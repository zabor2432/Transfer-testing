from dagster import Definitions, asset


class StupidModel:
    def __init__(self, epochs=10):
        self.epochs = epochs

    def train(self, x, y):
        pass

    def predict(self, x):
        return True


def mock_train_test_split(books):
    return books[:2], books[2:]


@asset
def augmented_books():
    return ["Book A", "Book B", "Book C"]


@asset
def ml_model(augmented_books):
    train_dataset, test_dataset = mock_train_test_split(augmented_books)
    model = StupidModel()
    model.train(train_dataset, test_dataset)
    return model


@asset
def test_model(ml_model):
    score = ml_model.predict("Book D")
    return score


defs = Definitions(assets=[augmented_books, ml_model, test_model])
