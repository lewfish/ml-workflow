from pt.recog.models.mini import MINI, Mini


def make_model(model_type, input_shape):
    if model_type == MINI:
        model = Mini(input_shape)

    return model
