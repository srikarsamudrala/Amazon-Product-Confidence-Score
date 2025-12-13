import pandas as pd, numpy as np, json, os
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Dict
from multiprocessing import Pool, cpu_count #https://superfastpython.com/multiprocessing-pool-map #data processing done in batches+chunks

try:#sentiment analysis
    from textblob import TextBlob
    import nltk
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt', quiet=True)
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords', quiet=True)
except ImportError:
    print("Cannot access textblob"); TextBlob = None

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer#vader is for sentiment

import matplotlib.pyplot as plt#viz library 1       
import seaborn as sns#viz library 2
try: plt.style.use('seaborn-v0_8-darkgrid')
except: plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

def _process_row_static(args):#multiproc helper. rows run parallely. refer line 4 link
    row, analyzer = args

    if not ReviewProcessor._validate_row_static(row): return None
    asin = str(row['asin']).strip()
    rating = float(row['rating'])
    review_text = str(row['reviewText']).strip()
    sentiment_scores = analyzer.analyze(review_text)
    return (asin, rating, sentiment_scores)

class SentimentAnalyzer:#textblob + vader. huggingface disabled for now because it would take longest time. next step though
    def __init__(self, use_gpu=True):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.hf_available = False; self.hf_pipeline = None

    def _textblob_score(self, text):
        if TextBlob is None: return None
        if not text or pd.isna(text) or str(text).strip()=="": return 3.0
        try:
            blob = TextBlob(str(text))
            score = 3.0 + (blob.sentiment.polarity*2.0) #neutral = 3.0 = halfway of 1-5
            return max(1.0, min(5.0, score))
        except: return 3.0

    def _vader_score(self, text):
        if not text or pd.isna(text) or str(text).strip()=="": return 3.0
        try:
            compound = self.vader_analyzer.polarity_scores(str(text))['compound']
            score = 3.0 + (compound*2.0)
            return max(1.0, min(5.0, score))
        except: return 3.0

    def analyze(self, text):
        s={}; tb=self._textblob_score(text)
        if tb is not None: s['textblob_score']=tb
        s['vader_score']=self._vader_score(text)
        return s

class ReviewProcessor: #csv chunking+cleaning+aggregation of results
    @staticmethod #static = cant modify instance variables
    def _validate_row_static(row):
        cols=['reviewText','rating','asin']
        if not all(c in row.index for c in cols): return False
        if 'is_spam' in row.index:
            val=row['is_spam']
            if pd.isna(val): return False
            try:
                if int(val)==1: return False
            except: 
                if str(val).lower() in ['1','true','spam','yes']: return False
        if not str(row.get('reviewText','')).strip(): return False
        r=row.get('rating',None)
        if pd.isna(r): return False
        try:
            r=float(r)
            if r<1 or r>5: return False
        except: return False
        return True

    def __init__(self, input_file, output_dir="output", chunk_size=50000, use_gpu=True):
        self.input_file=input_file; self.output_dir=Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)#
        self.chunk_size=chunk_size
        self.sentiment_analyzer=SentimentAnalyzer(use_gpu)
        self.asin_data={}; self.column_mapping={}
        print(f"init ReviewProcessor\nfile: {input_file}\nchunksize: {chunk_size}")

    def _clean_text(self, text): return "" if pd.isna(text) else str(text).strip()
    
    
    
    def _process_chunk(self, chunk):
        total=len(chunk)
        print(f"processing chunk ({total:,})")
        rows=[(r,self.sentiment_analyzer) for _,r in chunk.iterrows()]
        num_workers=max(1,cpu_count()-1)
        valid=0
        for i,res in enumerate(Pool(num_workers).imap(_process_row_static, rows),1):
            if res is None: continue
            valid+=1; asin,rating,scores=res
            if asin not in self.asin_data:
                self.asin_data[asin]={'ratings':[],'sentiment_scores':{'textblob':[],'vader':[],'huggingface':[]}}
            self.asin_data[asin]['ratings'].append(rating)
            if 'textblob_score' in scores: self.asin_data[asin]['sentiment_scores']['textblob'].append(scores['textblob_score'])
            if 'vader_score' in scores: self.asin_data[asin]['sentiment_scores']['vader'].append(scores['vader_score'])
            if i % max(1,total//10)==0:
                print(f"  progress {i/total*100:.0f}% ({i}/{total})")
        print(f"done {valid:,} valid rows")

    def process(self):
        print("\nstarting processing")
        if not os.path.exists(self.input_file): raise FileNotFoundError(f"{self.input_file} not found")
        s=pd.read_csv(self.input_file,nrows=100)
        print("cols:",list(s.columns))
        for c in s.columns:
            cl=c.lower()
            if 'reviewtext' in cl: self.column_mapping['reviewText']=c
            if 'rating' in cl or cl=='overall': self.column_mapping['rating']=c
            if 'asin' in cl: self.column_mapping['asin']=c
            if 'spam' in cl or cl in ['class','is_spam']: self.column_mapping['is_spam']=c
        print("mapped:",self.column_mapping)
        total_chunks=sum(1 for _ in pd.read_csv(self.input_file,chunksize=self.chunk_size))
        ch=0; rows=0
        for chunk in pd.read_csv(self.input_file,chunksize=self.chunk_size,low_memory=False,
                                 dtype={self.column_mapping.get('asin','asin'):str}):
            ch+=1; rows+=len(chunk)
            chunk=chunk.rename(columns={v:k for k,v in self.column_mapping.items()})
            self._process_chunk(chunk)
            print(f"overall {(ch/total_chunks)*100:.1f}% ({ch}/{total_chunks}) chunks, {rows:,} rows")
        print(f"done {ch} chunks, {rows:,} rows total\nunique asins: {len(self.asin_data):,}")

        print("aggregating...")
        results=[]
        for asin,data in self.asin_data.items():
            if not data['ratings']: continue
            avg_rating=np.mean(data['ratings'])
            sentiment_averages={}; all_scores=[]
            for m in ['textblob','vader']:
                sc=data['sentiment_scores'][m]
                if sc: sentiment_averages[m]=np.mean(sc); all_scores+=sc
            avg_sent=np.mean(all_scores) if all_scores else 3.0
            conf=0.65*avg_rating+0.35*avg_sent
            results.append({
                'asin':asin,'average_amazon_rating':round(avg_rating,3),
                'average_sentiment_score':round(avg_sent,3),
                'product_confidence_score':round(conf,3),
                'num_reviews':len(data['ratings']),
                'textblob_avg':round(sentiment_averages.get('textblob',0),3),
                'vader_avg':round(sentiment_averages.get('vader',0),3),
                'huggingface_avg':0.0
            })
        df=pd.DataFrame(results)
        print(f"aggregated {len(df):,} asins")
        return df

    def save_results(self, df, format='csv'):
        base=self.output_dir/"product_confidence_results"
        if format in ['csv','both']:
            p=base.with_suffix('.csv'); df.to_csv(p,index=False)
            print("saved csv:",p)
        if format in ['json','both']:
            p=base.with_suffix('.json'); df.to_json(p,orient='records',indent=2)
            print("saved json:",p)

    def generate_visualizations(self, df):
        if df.empty: print("no data to visualize"); return
        fig,ax=plt.subplots(2,2,figsize=(14,10))
        fig.suptitle("Amazon Product Confidence Analysis",fontsize=16,fontweight='bold')
        ax[0,0].hist(df['product_confidence_score'],bins=30,alpha=.7,color='steelblue',edgecolor='black')
        ax[0,0].set_title('Confidence Distribution'); ax[0,0].set_xlabel('Score')
        sc=ax[0,1].scatter(df['average_amazon_rating'],df['product_confidence_score'],
                           c=df['num_reviews'],cmap='viridis',alpha=.6)
        plt.colorbar(sc,ax=ax[0,1],label='Num Reviews')
        top=df.nlargest(min(15,len(df)),'product_confidence_score')
        ax[1,0].barh(range(len(top)),top['product_confidence_score'],color='teal')
        ax[1,0].set_yticks(range(len(top))); ax[1,0].set_yticklabels([a[:10]+"…" for a in top['asin']])
        ax[1,0].set_title('Top ASINs by Confidence')
        ax[1,1].scatter(df['average_amazon_rating'],df['average_sentiment_score'],alpha=.6,color='coral')
        ax[1,1].plot([1,5],[1,5],'k--',alpha=.5); ax[1,1].set_title('Rating vs Sentiment')
        plt.tight_layout(rect=[0,0,1,0.96])
        path=self.output_dir/"analysis_visualizations.png"
        plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
        print("visuals saved:",path)
        
def main():
    INPUT_FILE="cleaned_dataset.csv"; OUTPUT_DIR="output"; CHUNK_SIZE=60000
    USE_GPU=True; GEN_VIZ=True; OUT_FMT='both'
    print("="*60); print("Amazon Review Dataset Processor"); print("="*60)
    print(f"\nfile: {INPUT_FILE}\nchunk: {CHUNK_SIZE}\nGPU: {USE_GPU}\nvisuals: {GEN_VIZ}\n")
    p=ReviewProcessor(INPUT_FILE,OUTPUT_DIR,CHUNK_SIZE,USE_GPU)
    df=p.process(); p.save_results(df,OUT_FMT)
    if GEN_VIZ: p.generate_visualizations(df)
    print("\nProcessing done.")

if __name__=="__main__": main()