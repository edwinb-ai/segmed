from utils import utils


def extract_data():

    x, y = utils.extract_data(
        "tests/example_dataset/images_prepped_train/*.png",
        "tests/example_dataset/annotations_prepped_train/*.png",
    )
    print(x.shape)
    print(y.shape)

    expected_shape_x = (5, None, None, 3)

if __name__ == "__main__":
    extract_data()
