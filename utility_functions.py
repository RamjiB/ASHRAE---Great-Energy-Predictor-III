import seaborn as sns


def distribution_plots(grouped):
    keys = grouped.groups.keys()
    for i, b_id in enumerate(keys):
        g = grouped.get_group(b_id)
        try:
            f = sns.distplot(g['meter_reading']).set_title(str(b_id))
            f.get_figure().savefig('distribution_plots/building_id_' + str(b_id) + '.png')
            f.get_figure().clf()
        except ValueError:
            print('missed to save', b_id)

        if i % 100 == 0: print("saved {} images".format(i))

    print('Plots saved Successfully')
