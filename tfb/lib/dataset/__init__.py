def get_dataset(args, oracle):
    if args.task == "tfbind":
        from lib.dataset.regression import TFBind8Dataset
        return TFBind8Dataset(args, oracle)
