from progress.bar import Bar
import seaborn as sns


def distribution_plots(grouped):
    bar = Bar('saving distribution plots', max=len(grouped.groups.keys()))
    for i, b_id in enumerate(grouped.groups.keys()):
        g = grouped.get_group(b_id)
        try:
            f = sns.distplot(g['meter_reading'])
            f.get_figure().savefig('distribution_plots/building_id_' + str(b_id) + '.png')
        except ValueError:
            print ('missed to save',b_id)
        bar.next()
    bar.finish()
