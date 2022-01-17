import matplotlib.pyplot as plt

names = ["base-random", "base-english", "base-multilingual", "AI-Base", "KB-BERT", "AI-Large"]
eval_f1 = [0.9537, 0.965, 0.9699, 0.976, 0.9791, 0.9813]
test_f1 = [0.9539, 0.9637, 0.9685, 0.9748, 0.9781, 0.9804]


def main():
    y = eval_f1
    plt.bar(names, y, width=0.4, edgecolor="black")
    plt.xticks(rotation=45)
    plt.locator_params(nbins=4)
    # fig, ax = plt.subplots()
    for i, v in enumerate(y):
        plt.text(i - 0.1, v + 0.001, str(v), color='black')
        # plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    plt.ylabel("F1")
    plt.title("Swedish Reviews Classification - Eval F1")
    plt.ylim([0.95, 0.985])
    plt.show()


if __name__ == '__main__':
    main()
