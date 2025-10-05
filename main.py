import argparse
import data_collection
import advanced_feature_engineering
import meta_model_trainer
import advanced_options_trader
import config

def main():
    """
    Main control function to orchestrate the trading bot's operations.
    Uses command-line arguments to decide which action to perform.
    """
    parser = argparse.ArgumentParser(description="Advanced AI Trading Bot Control Panel")
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['setup', 'trade_options', 'retrain'],
        help="The action to perform: 'setup' (run full data pipeline and train model), "
             "'trade_options' (run the live options trading bot), "
             "'retrain' (run the model training process only)."
    )
    args = parser.parse_args()

    if args.action == 'setup':
        print("--- Running Full Setup: Data -> Features -> Model ---")
        data_collection.run_collection()
        advanced_feature_engineering.run_feature_engineering()
        meta_model_trainer.run_model_training()
        print("\n--- Full Setup is Complete! You are now ready to trade. ---")
        print(f"Run 'python main.py --action trade_options' to start the bot.")

    elif args.action == 'trade_options':
        advanced_options_trader.run_trader()

    elif args.action == 'retrain':
        print("--- Re-training the Ultimate AI Model ---")
        meta_model_trainer.run_model_training()
        print("\n--- Model Re-training Complete! ---")

if __name__ == '__main__':
    # A simple check to ensure API keys are set before running anything
    if 'YOUR_API_KEY' in config.API_KEY or 'YOUR_SECRET_KEY' in config.SECRET_KEY:
        print("\n!!! WARNING: Alpaca API keys are not configured in 'config.py'. Please edit the file. !!!\n")
    else:
        main()

