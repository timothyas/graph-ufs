from prototypes.gefs.postprocess_inference import main

from config import GEFSForecastEvaluator, GEFSDeviationEvaluator

if __name__ == "__main__":
    main(GEFSForecastEvaluator)
    #main(GEFSDeviationEvaluator)
