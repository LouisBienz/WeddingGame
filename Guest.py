class Guest:
    def __init__(self, data):
        self.name = f"{data[0]} {data[1]}"
        self.answers = [int(answer) for answer in data[2:]]