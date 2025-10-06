import argparse
import threading
import data_collection
import advanced_feature_engineering
import meta_model_trainer
import advanced_options_trader

def run_full_setup():
    """Runs the complete data pipeline and model training process."""
    print("\n--- Running Full Setup: Data -> Features -> Model ---")
    data_collection.run_collection()
    advanced_feature_engineering.run_feature_engineering()
    meta_model_trainer.run_training()
    print("\n--- Full Setup is Complete! You are now ready to trade. ---")

def main():
    """Main function to parse arguments and run the specified action."""
    parser = argparse.ArgumentParser(description="AITradePro Master Control Script")
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['setup', 'trade_options', 'retrain'],
        help="The action to perform."
    )
    
    args = parser.parse_args()

    if args.action == 'setup' or args.action == 'retrain':
        run_full_setup()
    elif args.action == 'trade_options':
        # This is now primarily handled by the API server, 
        # but this allows for local, standalone testing.
        print("Starting trader in standalone mode...")
        stop_event = threading.Event()
        try:
            advanced_options_trader.run_trader(stop_event)
        except KeyboardInterrupt:
            print("\nStandalone trader stopped by user.")
            stop_event.set()
    else:
        print(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main()

