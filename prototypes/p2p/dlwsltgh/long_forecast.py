from prototypes.p2p.long_forecast import run_long_forecast
from config import P2PEvaluator as Emulator

if __name__ == "__main__":
    run_long_forecast(Emulator, tf="2019-06-30T21")
