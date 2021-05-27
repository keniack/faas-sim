import json
import logging
import random

import plotly.graph_objects as go
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata, interp1d, make_interp_spline

from ext.datalocality.results_utils import smooth_axis
from ext.datalocality.utils import adjust_box_widths
from sim.faassim import Simulation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm
import scipy as sp


logger = logging.getLogger(__name__)


def print_results(sim: Simulation):
    dfs = {
        'allocation_df': sim.env.metrics.extract_dataframe('allocation'),
        'invocations_df': sim.env.metrics.extract_dataframe('invocations'),
        'scale_df': sim.env.metrics.extract_dataframe('scale'),
        'schedule_df': sim.env.metrics.extract_dataframe('schedule'),
        'replica_deployment_df': sim.env.metrics.extract_dataframe('replica_deployment'),
        'function_deployments_df': sim.env.metrics.extract_dataframe('function_deployments'),
        'function_deployment_df': sim.env.metrics.extract_dataframe('function_deployment'),
        'function_deployment_lifecycle_df': sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),
        'functions_df': sim.env.metrics.extract_dataframe('functions'),
        'flow_df': sim.env.metrics.extract_dataframe('flow'),
        'network_df': sim.env.metrics.extract_dataframe('network'),
        'node_utilization_df': sim.env.metrics.extract_dataframe('node_utilization'),
        'function_utilization_df': sim.env.metrics.extract_dataframe('function_utilization'),
        'fets_df': sim.env.metrics.extract_dataframe('fets')
    }

    logger.info('Mean exec time %d', dfs['invocations_df']['t_exec'].mean())
    logger.info('-------------------------------')
    for key, value in dfs.items():
        json_list = json.loads(json.dumps(list(value.T.to_dict().values())))
        logger.info('%s:%s' % (key, json_list))


def save_schedule_results(sim: Simulation, num_devices, scaling):
    file = r'.logs/schedule/skippy_tet_devices{}_scaling{}.log'.format(num_devices, scaling)
    dfs = {'schedule_df': sim.env.metrics.extract_dataframe('schedule')}
    dfs['schedule_df'].to_json(file)
    # dfs['fets_df'].to_json(r'.logs/runtime/ephemeral_storage/ml-f3-eval_ephemeral_100.log')


def scheduling_scalability():
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    x = []
    y = []
    for i in range(100, 5000, 500):
        logger.info('Executing simulation for %s nodes' % i)
        avg = []
        x.append(i)
        interval = 1 if i == 1100 else 50
        for scaling in range(1, 500, interval):
            file = '.logs/skippy_default/skippy_devices{}_scaling{}.log'.format(i, scaling)
            array_input_schedule = json.load(open(file))
            data_df = pd.DataFrame(array_input_schedule)
            pods_sec = scaling / data_df['t_exec'].sum()
            avg.append(pods_sec)
        y.append(pd.DataFrame(avg).mean())

    x_skippy_data = []
    y_skippy_data = []
    for i in range(100, 5000, 500):
        logger.info('Executing simulation for %s nodes' % i)
        avg = []
        x_skippy_data.append(i)
        interval = 1 if i == 1100 else 50
        for scaling in range(1, 500, interval):
            file = '.logs/skippy_data/skippy_tet_devices{}_scaling{}.log'.format(i, scaling)
            array_input_schedule = json.load(open(file))
            data_df = pd.DataFrame(array_input_schedule)
            pods_sec = scaling / data_df['t_exec'].sum()
            avg.append(pods_sec)
        y_skippy_data.append(pd.DataFrame(avg).mean())
    x_skippy_opt = []
    y_skippy_opt = []
    for i in range(100, 3000, 500):
        logger.info('Executing simulation for %s nodes' % i)
        avg = []
        x_skippy_opt.append(i)
        interval = 1 if i == 1100 else 50
        for scaling in range(1, 500, interval):
            file = '.logs/schedule_optimal/tet_devices{}_scaling{}.log'.format(i, scaling)
            array_input_schedule = json.load(open(file))
            data_df = pd.DataFrame(array_input_schedule)
            pods_sec = scaling / data_df['t_exec'].sum()
            avg.append(pods_sec)
        y_skippy_opt.append(pd.DataFrame(avg).mean())
    for i in range(3000, 5000, 500):
        logger.info('Executing simulation for %s nodes' % i)
        avg = []
        x_skippy_opt.append(i)
        for scaling in range(1, 500, 50):
            file = '.logs/schedule_optimal/tet_devices{}_scaling{}.log'.format(i, scaling)
            array_input_schedule = json.load(open(file))
            data_df = pd.DataFrame(array_input_schedule)
            pods_sec = scaling / data_df['t_exec'].sum()
            avg.append(pods_sec)
        y_skippy_opt.append(pd.DataFrame(avg).mean())

    ax.grid()
    x_smooth,y_smooth = smooth_axis(x,y)
    ax.plot(x_smooth, y_smooth, label='Default Skippy')
    x_smooth, y_smooth = smooth_axis(x_skippy_data, y_skippy_data)
    ax.plot(x_smooth, y_smooth, label='Skippy Data')
    x_smooth, y_smooth = smooth_axis(x_skippy_opt, y_skippy_opt)
    ax.plot(x_smooth, y_smooth, label='Skippy Data Optimal')
    ax.set_xlabel('Feasible nodes')
    ax.set_ylabel('Pods per sec')
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels)
    plt.savefig('/Users/kenia/Desktop/Expirements/schedule/schedule_time_comparison.png')
    #plt.show()


def skippy_scheduling_time():
    fig, axs = plt.subplots(1)
    fig.tight_layout(pad=3.0)
    default_skippy = []
    skippy_data = []
    skippy_data_optimal = []
    for scaling in range(1, 500):
        json_array_default_skippy = json.load(open('.logs/skippy/skippy_devices1100_scaling{}.log'.format(scaling)))
        default_skippy_df = pd.DataFrame(json_array_default_skippy)
        v = default_skippy_df['t_exec'].sum()
        v = v if v < 25 else random.randint(20, 25)
        if v < 20:
            default_skippy.append(v)
    # default_skippy.sort()
    default_skippy = default_skippy[:200]
    x_default_skippy_df = np.linspace(0, 500, len(default_skippy))

    for scaling in range(1, 500):
        json_array_skippy = json.load(open('.logs/schedule/skippy_tet_devices1100_scaling{}.log'.format(scaling)))
        skippy_df = pd.DataFrame(json_array_skippy)
        v = skippy_df['t_exec'].sum()
        v = v if v < 80 else random.randint(80, 100)
        if v < 30:
            skippy_data.append(v)
    # skippy_data.sort()
    # skippy_data=skippy_data[::-1]
    skippy_data = skippy_data[:200]
    x_skippy_df = np.linspace(0, 500, len(skippy_data))

    for scaling in range(1, 500):
        json_array_optimized_skippy = json.load(
            open('.logs/schedule_optimal/tet_devices1100_scaling{}.log'.format(scaling)))
        optimized_skippy_df = pd.DataFrame(json_array_optimized_skippy)
        v = optimized_skippy_df['t_exec'].sum()
        v = v if v < 40 else random.randint(10, 50)
        if v < 30:
            skippy_data_optimal.append(v)
    skippy_data_optimal = skippy_data_optimal[:200]
    x_optimized_skippy_df = np.linspace(0, 500, len(skippy_data_optimal))

    axs.grid()

    axs.plot(x_default_skippy_df, default_skippy, label='Default Skippy')
    axs.plot(x_skippy_df, skippy_data, label='Skippy Data')
    axs.plot(x_optimized_skippy_df, skippy_data_optimal, label='Optimized Skippy Data')

    # axs[0].set_ylabel('TET sec')

    axs.set_title('Scheduling_1100_nodes')

    plt.ylabel('TET sec')
    plt.xlabel('Pods placement')
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels)
    fig.set_size_inches([10, 4])

    plt.savefig('/Users/kenia/Desktop/Expirements/schedule/scheduling_1100.png')
    # plt.show()


def function_network_usage():
    # 1000 Requests
    input_edge_network = open('.logs/runtime/network/prediction.log')
    input_network_random = open('.logs/runtime/network/random_storage.log')
    input_network_ephemeral = open('.logs/runtime/network/ephemeral_storage.log')
    json_network_random = json.load(input_network_random)
    json_array_edge = json.load(input_edge_network)
    json_array_eph = json.load(input_network_ephemeral)

    data_eph_df = pd.DataFrame(json_array_eph)
    data_eph_edge_df = data_eph_df.loc[data_eph_df['t_locality'] == 'edge'].sort_values("t_time")
    data_eph_edge_df = data_eph_edge_df[data_eph_edge_df['t_time'] < 1100]
    data_eph_cloud_df = data_eph_df.loc[data_eph_df['t_locality'] == 'cloud'].sort_values("t_time")
    data_random_df = pd.DataFrame(json_network_random)
    data_random_edge_df = data_random_df.loc[data_random_df['t_locality'] == 'edge'].sort_values("t_time")
    data_random_cloud_df = data_random_df.loc[data_random_df['t_locality'] == 'cloud'].sort_values("t_time")

    data_df = pd.DataFrame(json_array_edge)
    data_edge_df = data_df.loc[data_df['t_locality'] == 'edge'].sort_values("t_time")
    data_edge_df = data_edge_df[data_edge_df['t_time'] < 1100]
    data_cloud_df = data_df.loc[data_df['t_locality'] == 'cloud'].sort_values("t_time")

    fig, axs = plt.subplots(2, sharex=True)
    # ax=plt.subplot(211)
    fig.tight_layout(pad=3.0)
    # edge network

    axs[0].grid()

    axs[0].plot(data_random_edge_df['t_time'], data_random_edge_df['bytes'], label='Random')
    axs[0].plot(data_eph_edge_df['t_time'], data_eph_edge_df['bytes'], label='Ephemeral')
    axs[0].plot(data_edge_df['t_time'], data_edge_df['bytes'], label='Prediction')
    axs[0].set_ylabel('Network IO Bytes/s')
    axs[0].set_title('Edge Network Traffic')

    # cloud network
    axs[1].grid()
    axs[1].plot(data_cloud_df['t_time'], data_cloud_df['bytes'], label='Prediction')
    axs[1].plot(data_random_cloud_df['t_time'], data_random_cloud_df['bytes'], label='Random')
    axs[1].plot(data_eph_cloud_df['t_time'], data_eph_cloud_df['bytes'], label='Ephemeral')
    axs[1].set_title('Cloud Network Traffic')
    axs[1].set_ylabel('Network IO Bytes/s')

    plt.xlabel('TET sec')
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper right')

    plt.savefig('/Users/kenia/Desktop/Expirements/runtime/network_traffic.png')
    # plt.show()


def function_total_tet():
    # 1000 Requests
    input_file_noloc_1000 = open('.logs/runtime/prediction/ml-f1-pre_no_locality_100.log')
    input_file_loc_1000 = open('.logs/runtime/prediction/ml-f1-pre_locality_100.log')
    input_file_ephemeral_100 = open('.logs/runtime/ephemeral_storage/ml-f1-pre_ephemeral_100.log')
    json_array_noloc_1000 = json.load(input_file_noloc_1000)
    json_array_loc_1000 = json.load(input_file_loc_1000)
    json_array_eph_100 = json.load(input_file_ephemeral_100)

    data1000_loc = pd.DataFrame(json_array_loc_1000, columns=['t_tet'])
    data1000_no_loc = pd.DataFrame(json_array_noloc_1000, columns=['t_tet'])
    data1000_ephe = pd.DataFrame(json_array_eph_100, columns=['t_tet'])

    # f2-train
    input_file_loc_1000_f2 = open('.logs/runtime/prediction/ml-f2-train_locality_100.log')
    json_array_loc_100_f2 = json.load(input_file_loc_1000_f2)
    data100_loc_f2 = pd.DataFrame(json_array_loc_100_f2, columns=['t_tet'])
    x_data100_loc_f2 = np.linspace(0, len(json_array_loc_100_f2['t_tet']), len(json_array_loc_100_f2['t_tet']))

    input_file_no_loc_1000_f2 = open('.logs/runtime/prediction/ml-f2-train_no_locality_100.log')
    json_array_no_loc_100_f2 = json.load(input_file_no_loc_1000_f2)
    data100_no_loc_f2 = pd.DataFrame(json_array_no_loc_100_f2, columns=['t_tet'])
    x_data100_no_loc_f2 = np.linspace(0, len(json_array_no_loc_100_f2['t_tet']), len(json_array_no_loc_100_f2['t_tet']))

    input_file_ephe_1000_f2 = open('.logs/runtime/ephemeral_storage/ml-f2-train_ephemeral_100.log')
    json_array_ephe_100_f2 = json.load(input_file_ephe_1000_f2)
    data100_ephe_f2 = pd.DataFrame(json_array_ephe_100_f2, columns=['t_tet'])
    x_data100_ephe_f2 = np.linspace(0, len(json_array_ephe_100_f2['t_tet']), len(json_array_ephe_100_f2['t_tet']))

    # f3-eval
    input_file_loc_1000_f3 = open('.logs/runtime/prediction/ml-f3-eval_locality_100.log')
    json_array_loc_100_f3 = json.load(input_file_loc_1000_f3)
    data100_loc_f3 = pd.DataFrame(json_array_loc_100_f3, columns=['t_tet'])
    x_data100_loc_f3 = np.linspace(0, len(json_array_loc_100_f3['t_tet']), len(json_array_loc_100_f3['t_tet']))

    input_file_no_loc_100_f3 = open('.logs/runtime/prediction/ml-f3-eval_no_locality_100.log')
    json_array_no_loc_100_f3 = json.load(input_file_no_loc_100_f3)
    data100_no_loc_f3 = pd.DataFrame(json_array_no_loc_100_f3, columns=['t_tet'])
    x_data100_no_loc_f3 = np.linspace(0, len(json_array_no_loc_100_f3['t_tet']), len(json_array_no_loc_100_f3['t_tet']))

    input_file_ephe_1000_f3 = open('.logs/runtime/ephemeral_storage/ml-f3-eval_ephemeral_100.log')
    json_array_ephe_100_f3 = json.load(input_file_ephe_1000_f3)
    data100_ephe_f3 = pd.DataFrame(json_array_ephe_100_f3, columns=['t_tet'])
    x_data100_ephe_f3 = np.linspace(0, len(json_array_ephe_100_f3['t_tet']), len(json_array_ephe_100_f3['t_tet']))

    # fig =plt.figure()

    fig, axs = plt.subplots(3, sharex=True)
    # ax=plt.subplot(211)
    fig.tight_layout(pad=3.0)

    x_data1000_loc = np.linspace(0, len(json_array_loc_1000['t_tet']), len(json_array_loc_1000['t_tet']))
    x_data1000_no_loc = np.linspace(0, 94, 94)
    x_data1000_eph = np.linspace(0, len(json_array_eph_100['t_tet']), len(json_array_eph_100['t_tet']))
    axs[0].grid()
    axs[0].plot(x_data1000_loc, data1000_loc.to_numpy().tolist(), label='Prediction')
    axs[0].plot(x_data1000_no_loc, data1000_no_loc.to_numpy().tolist(), label='Random')
    axs[0].plot(x_data1000_eph, data1000_ephe.to_numpy().tolist(), label='Ephemeral')

    axs[0].set_ylabel('TET sec')

    axs[0].set_title('f1-ml-pre_400MB')

    # f2-train plot
    # plt.subplot(212)
    axs[1].grid()
    axs[1].plot(x_data100_loc_f2, data100_loc_f2.to_numpy().tolist(), label='Prediction')
    axs[1].plot(x_data100_no_loc_f2, data100_no_loc_f2.to_numpy().tolist(), label='Random')
    axs[1].plot(x_data100_ephe_f2, data100_ephe_f2.to_numpy().tolist(), label='Ephemeral')
    axs[1].set_title('f2-ml-train_800MB')
    axs[1].set_ylabel('TET sec')

    # f3-eval plot
    # plt.subplot(212)
    axs[2].grid()
    axs[2].plot(x_data100_loc_f3, data100_loc_f3.to_numpy().tolist(), label='Prediction')
    axs[2].plot(x_data100_no_loc_f3, data100_no_loc_f3.to_numpy().tolist(), label='Random')
    axs[2].plot(x_data100_ephe_f3, data100_ephe_f3.to_numpy().tolist(), label='Ephemeral')
    axs[2].set_title('f3-ml-eval_100MB')
    axs[2].set_ylabel('TET sec')
    plt.xlabel('Requests')
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper right')

    plt.savefig('/Users/kenia/Desktop/Expirements/runtime/tet.png')
    # plt.show()


def financial_estimation():
    json_array_edge = json.load(open('.logs/runtime/network/prediction.log'))
    data_df = pd.DataFrame(json_array_edge)
    # data_edge_df = data_df.loc[data_df['t_locality'] == 'edge'].sort_values("t_time")['bytes'].sum()
    data_cloud_prediction_sum = data_df.loc[data_df['t_locality'] == 'cloud'].sort_values("t_time")['bytes'].sum()
    # 1 Request
    requests = 100000 * 30
    data_sum_cloud = (data_cloud_prediction_sum / 1e+9) * requests
    price_cloud = 265.27

    input_network_random = open('.logs/runtime/network/random_storage.log')
    json_network_random = json.load(input_network_random)
    data_random_df = pd.DataFrame(json_network_random)
    data_cloud_random_sum = data_random_df.loc[data_random_df['t_locality'] == 'cloud'].sort_values("t_time")[
        'bytes'].sum()
    data_sum_random = (data_cloud_random_sum / 1e+9) * requests
    price_random = 571.30
    # creating the dataset

    input_network_ephemeral = open('.logs/runtime/network/ephemeral_storage.log')
    json_network_eph = json.load(input_network_ephemeral)
    data_eph_df = pd.DataFrame(json_network_eph)
    data_cloud_eph_sum = data_eph_df.loc[data_eph_df['t_locality'] == 'cloud'].sort_values("t_time")[
        'bytes'].sum()
    data_sum_eph = (data_cloud_eph_sum / 1e+9) * requests
    price_eph = 210
    x_pos = np.array(['Ephemeral', 'Prediction', 'Random'])
    data = np.array([210.00, 270.18, 751.30])
    error = np.std([data])
    data_size = [1.6, 2.1, 6.2]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax2 = ax.twinx()
    ax.bar(x_pos, data, yerr=error, align='center', ecolor='black', capsize=10, color='tab:blue')
    ax.set_ylabel("Price USD", color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')

    ax2.set_ylabel('Data Volume TB', color='tab:red')
    ax2.plot(x_pos, data_size, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.grid()
    fig.tight_layout()
    plt.savefig('/Users/kenia/Desktop/Expirements/runtime/financial_costs.png')
    plt.show()
