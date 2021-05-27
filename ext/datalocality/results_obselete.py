import json
import logging

from matplotlib import cm
from networkx.drawing.tests.test_pylab import plt
from scipy.interpolate import griddata
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def scheduling_tet3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    for i in range(100, 5000, 500):
        logger.info('Executing simulation for %s nodes' % i)
        for scaling in range(1, 500, 50):
            file = '.logs/skippy/skippy_devices{}_scaling{}.log'.format(i, scaling)
            array_input_schedule = json.load(open(file))
            data_df = pd.DataFrame(array_input_schedule)
            x.append(scaling)
            y.append(i)
            v = data_df['t_exec'].sum()
            z.append(v)

    xyz = {'x': x, 'y': y, 'z': z}
    df = pd.DataFrame(xyz, index=range(len(xyz['x'])))

    # re-create the 2D-arrays
    x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()) * 10)
    y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()) * 10)
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='cubic')

    z2.sort(axis=1)

    surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    # z.sort()
    # ax.scatter(x, y, z)
    ax.set_zlim3d(0, 29)
    ax.set_ylabel('Feasible nodes')
    ax.set_xlabel('Pods placements')
    ax.set_zlabel('TET sec')
    # ax.set_title('Default Skippy')
    ax.view_init(15, -120)

    plt.savefig('/Users/kenia/Desktop/Expirements/schedule/skippy_default.png')
    # plt.show()

def print_ml_f1_box_plot():
    # 10 Requests
    input_file_noloc = open('.logs/runtime/prediction/ml-f1-pre_no_locality_10.log')
    input_file_loc = open('.logs/runtime/prediction/ml-f1-pre_locality_10.log')
    json_array_noloc = json.load(input_file_noloc)
    json_array_loc = json.load(input_file_loc)

    # 100 Requests
    input_file_noloc_100 = open('.logs/runtime/prediction/ml-f1-pre_no_locality_100.log')
    input_file_loc_100 = open('.logs/runtime/prediction/ml-f1-pre_locality_100.log')
    json_array_noloc_100 = json.load(input_file_noloc_100)
    json_array_loc_100 = json.load(input_file_loc_100)

    # 1000 Requests
    input_file_noloc_1000 = open('.logs/runtime/prediction/ml-f1-pre_no_locality_1000.log')
    input_file_loc_1000 = open('.logs/runtime/prediction/ml-f1-pre_locality_1000.log')
    json_array_noloc_1000 = json.load(input_file_noloc_1000)
    json_array_loc_1000 = json.load(input_file_loc_1000)

    data10_loc = pd.DataFrame(json_array_loc, columns=['t_tet']).assign(Requests=10, Locality='Prediction')
    data10_no_loc = pd.DataFrame(json_array_noloc, columns=['t_tet']).assign(Requests=10, Locality='Random')

    data100_loc = pd.DataFrame(json_array_loc_100, columns=['t_tet']).assign(Requests=100, Locality='Prediction')
    data100_no_loc = pd.DataFrame(json_array_noloc_100, columns=['t_tet']).assign(Requests=100, Locality='Random')

    data1000_loc = pd.DataFrame(json_array_loc_1000, columns=['t_tet']).assign(Requests=1000, Locality='Prediction')
    data1000_no_loc = pd.DataFrame(json_array_noloc_1000, columns=['t_tet']).assign(Requests=1000, Locality='Random')

    cdf = pd.concat([data10_no_loc, data10_loc, data100_no_loc, data100_loc, data1000_no_loc, data1000_loc])
    mdf = pd.melt(cdf, id_vars=['Requests', 'Locality'])
    # print(mdf.head())

    fig = plt.figure()

    plt.grid()
    plt.title('f1-ml-pre')

    ax = sns.boxplot(x="Requests", y="value", hue="Locality", data=mdf, width=0.6)
    adjust_box_widths(fig, 0.9)
    ax.set_xlim(-0.5, 3)
    ax.set_ylabel('TET sec')
    ax.legend(loc='lower right')
    plt.show()
    # plt.savefig('/Users/kenia/Desktop/Expirements/runtime/ml1_400.png')
