from prototypes.gefs.inference import inference
from config import (
    GEFSForecastEvaluator as ForecastEvaluator,
    GEFSDeviationEvaluator as DeviationEvaluator,
)

if __name__ == "__main__":
    inference(ForecastEvaluator)
    inference(DeviationEvaluator)
