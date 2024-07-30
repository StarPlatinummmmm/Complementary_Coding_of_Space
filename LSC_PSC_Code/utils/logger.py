import os


class Logger:
    def __init__(self, path, filename):
        # path = os.path.join('./exp_result/', path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.filepath = os.path.join(path, filename)
        return

    def log(self, dic: dict):
        # Check if file exists to decide whether to write headers
        file_exists = os.path.exists(self.filepath)
        with open(self.filepath, 'a') as f:
            if not file_exists:
                # Write the header only once
                f.write(','.join(dic.keys()) + '\n')
            # Write the data, converting each value to string
            f.write(','.join(map(str, dic.values())) + '\n')
