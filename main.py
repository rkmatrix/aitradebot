import argparse
import data_collection
import advanced_feature_engineering
import meta_model_trainer
import advanced_options_trader

def run_full_setup():
    """Runs the complete data pipeline and model training process."""
    print("\n--- Running Full Setup: Data -> Features -> Model ---")
    data_collection.run_collection()
    advanced_feature_engineering.run_feature_engineering()
    # CRITICAL FIX: Call the correct function name 'run_training'
    meta_model_trainer.run_training()
    print("\n--- Full Setup is Complete! You are now ready to trade. ---")
    print("Run 'python main.py --action trade_options' to start the bot.")

def main():
    """Main function to parse arguments and run the specified action."""
    parser = argparse.ArgumentParser(description="AITradePro Master Control Script")
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['setup', 'trade_options', 'retrain'],
        help="The action to perform: 'setup' (run full data pipeline and training), "
             "'trade_options' (run the live options trading bot), "
             "'retrain' (alias for setup)."
    )
    
    args = parser.parse_args()

    if args.action == 'setup' or args.action == 'retrain':
        run_full_setup()
    elif args.action == 'trade_options':
        advanced_options_trader.run_trader()
    else:
        print(f"Unknown action: {args.action}")
        parser.print_help()

if __name__ == "__main__":
    main()

