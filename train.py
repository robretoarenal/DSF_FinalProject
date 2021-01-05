from utils import ScoresETL, ScoresTrain

def main():
    etl = ScoresETL(random_state=42)
    etl.etl_pipeline()
    features_full, results_full= etl.etl_pipeline()
    t = ScoresTrain(results_full=results_full,features_full=features_full,n_estimators=100).train()

if __name__ == "__main__":
    main()
