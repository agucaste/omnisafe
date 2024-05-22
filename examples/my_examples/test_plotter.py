from omnisafe.utils.my_plotter import MyPlotter

if __name__ == '__main__':
    import os
    import glob
    os.chdir('./runs/TRPOLagBinaryCritic-{SafetyPointCircle1-v0}/')
    print(os.getcwd())
    # experiments = {
    #     'safest': ['./' + file for file in glob.glob('051724mac/*')] + ['./' + file for file in glob.glob('*16-15-12-53')],
    #     'first_safe': ['./' + file for file in glob.glob('*05-16-15-15-37')],
    #     'safest & Bellman equality': ['./' + file for file in glob.glob('*05-16-15-16-3*')]
    # }

    # Different forms of safety gae.
    # safety_gae = ['seed-999-2024-05-20-16-18-09', 'seed-777-2024-05-20-16-18-16', 'seed-123-2024-05-20-16-18-02',
    #                    'seed-009-2024-05-20-16-17-52']
    # experiments = {
    #     'safety-gae': ['./' + exp for exp in safety_gae]
    # }

    # 05/22/24: Analysis of different forms of safety_idx (used to estimate v(.)) combined with Safety-GAE
    experiments = {
        'max': ['./' + fp for fp in glob.glob('*-2024-05-21-14-36-*')],
        'min': ['./' + fp for fp in glob.glob('*-2024-05-21-14-41-2*')],
        'avg': ['./' + fp for fp in glob.glob('*-2024-05-21-14-46-*')]
    }
    print(f' experiments = {experiments}')

    plotter = MyPlotter()
    plotter.make_plots(experiments=experiments, smooth=5,)
                     

