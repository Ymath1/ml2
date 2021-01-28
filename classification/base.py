

class Ensambler:

    def __init__(self):
        pass

    @staticmethod
    def run(list_of_models, test_generator, method='mode'):
        results = []
        if method == 'mode':
            for model in list_of_models:
                results.append(model.predict(gen_data=test_generator))
            for predictions in results[1:]:
                for i, prediction in enumerate(predictions):
                    results[0][i] = [sum(x) for x in zip(results[0][i], prediction)]

            ensambled = [[1 if max(small_list) == x else 0 for x in small_list] for small_list in results[0]]
            acc = sum([0 if x.index(max(x)) != y else 1 for x, y in zip(ensambled, test_generator.classes)])/len(test_generator.classes)

            print(f'Accuracy: {acc}')
            return ensambled


class ImageClassifier:

    def __init__(self):
        self.history = None
        self.evaluation = None

    def run(self):
        self.build()
        self.compile()

    def compile(self):
        raise NotImplementedError('Method must be override in derived classes')

    def build(self):
        raise NotImplementedError('Method must be override in derived classes')

    def fit(self):
        raise NotImplementedError('Method must be override in derived classes')

    def evaluate(self):
        raise NotImplementedError('Method must be override in derived classes')

    def predict(self):
        raise NotImplementedError('Method must be override in derived classes')

    def read_data(self):
        raise NotImplementedError('Method must be override in derived classes')
