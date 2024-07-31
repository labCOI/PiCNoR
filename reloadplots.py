from PlotViewer import PlotViewer
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address') 
    args = parser.parse_args()

    viewer = PlotViewer("Reloaded Plot")
    viewer.load_configuration(args.address)
    viewer.mainloop()