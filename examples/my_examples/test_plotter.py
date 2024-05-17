from omnisafe.utils.my_plotter import MyPlotter

if __name__ == '__main__':
    import os
    os.chdir('/Users/agu/PycharmProjects/omnisafe/examples/my_examples/runs/TRPOLagBinaryCritic-{SafetyPointCircle1-v0}')
    experiments = {
        'safest': ['./seed-000-2024-05-16-14-53-20', './seed-001-2024-05-16-14-53-24', './seed-999-2024-05-16-14-53-28'],
        'redudant': ['./seed-000-2024-05-16-14-53-20']
    }
    plotter = MyPlotter()
    plotter.make_plots(experiments=experiments)

