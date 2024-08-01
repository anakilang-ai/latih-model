def main():
    num = 'dataset-kelas'
    model_name = 'facebook/bart-base'
    output_dir = f'./result/results_coba{num}-{epoch}-{batch_size}'

    # Load and prepare dataset
    df = load_and_filter_dataset(f'{num}.csv')
    train_df, test_df = split_dataset(df)

    tokenizer = BartTokenizer.from_pretrained(model_name)
    dataset_train, dataset_test = prepare_datasets(train_df, test_df, tokenizer)

    # Initialize model
    model = BartForConditionalGeneration.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training arguments
    epoch = 20
    batch_size = 10
    training_args = configure_training_args(output_dir, num_train_epochs=epoch, batch_size=batch_size)

    # Generation configuration
    generation_config = setup_generation_config()

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_bleu_metrics(eval_pred, tokenizer)
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model_save_path = f'model/bart_coba{num}-{epoch}-{batch_size}'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    generation_config.save_pretrained(model_save_path)

    # Evaluate model
    eval_results = trainer.evaluate()

    # Format and log evaluation results
    eval_results_summary = "\n".join([f"{key}: {value}" for key, value in eval_results.items()])
    
    # Print and log evaluation results
    print(f"Evaluation results:\n{eval_results_summary}")
    logging.info(f"Model saved to: {model_save_path}")
    logging.info("Evaluation results:\n%s", eval_results_summary)
    logging.info("------------------------------------------\n")

if __name__ == "__main__":
    main()
