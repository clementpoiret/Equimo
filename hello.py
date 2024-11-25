import numpy as np


def main():
    print("Hello from nix-python-devenv!")
    print("As you can see, C bindings are correct, because numpy works:")
    print("a =", a := np.array([1, 2, 3]))
    print("mean(a) =", a.mean())


if __name__ == "__main__":
    main()
