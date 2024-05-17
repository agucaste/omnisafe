from omnisafe.utils.my_plotter import MyPlotter

if __name__ == '__main__':
    import os
    import glob
    os.chdir('./runs/TRPOLagBinaryCritic-{SafetyPointCircle1-v0}/')
    print(os.getcwd())
    experiments = {
        'safest': ['./' + file for file in glob.glob('051724mac/*')] + ['./' + file for file in glob.glob('*16-15-12-53')],
        'first_safe': ['./' + file for file in glob.glob('*05-16-15-15-37')],
        'safest & Bellman equality': ['./' + file for file in glob.glob('*05-16-15-16-3*')]
    }
    plotter = MyPlotter()
    plotter.make_plots(experiments=experiments, smooth=5)

