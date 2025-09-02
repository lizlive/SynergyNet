from Neural_model_kits.models_simple import *
def get_model(args):
    if args.data.task == "regression":
        if args.model.name == "SynergyNet":
            model = SynergyNet_my_regression(args.model)

    return model