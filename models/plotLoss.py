import matplotlib.pyplot as plt

def myPersonalizedPlot(id, all_train_loss, all_val_loss, all_train_metrics, all_val_metrics):

    # yyy = [-0.28835272789001465, -0.2811172604560852, -0.2895078957080841, -0.28319093585014343, -0.28631356358528137, -0.28041180968284607, -0.28710946440696716, -0.29234224557876587, -0.28504782915115356, -0.2856665849685669]
    # zzz = [-0.29294028878211975, -0.2935180068016052, -0.2941957116127014, -0.2945469617843628, -0.29492291808128357, -0.2953047752380371, -0.2954985797405243, -0.29570063948631287, -0.2959194481372833, -0.29603174328804016]
    #
    mmm = []
    gmmm = []

    for i in all_train_metrics:
        for metric_name, metric_value in i.items():
            # summary += " Train {metric_name} {metric_value}".format(
            #     metric_name=metric_name, metric_value=metric_value)
            if metric_name == 'georisk_10':
                gmmm.append(metric_value)
            if metric_name == 'ndcg_10':
                mmm.append(metric_value)
            # mmm.append(metric_value)

    mmmVal = []
    gmmmVal = []
    for i in all_val_metrics:
        for metric_name, metric_value in i.items():
            # summary += " Train {metric_name} {metric_value}".format(
            #     metric_name=metric_name, metric_value=metric_value)
            if metric_name == 'georisk_10':
                gmmmVal.append(metric_value)
            if metric_name == 'ndcg_10':
                mmmVal.append(metric_value)

    # id = "10_web"
    num_epochs = len(all_val_loss)
    # plt.plot(range(num_epochs), yyy)
    plt.plot(range(num_epochs), all_train_loss, label="Loss - Train")
    plt.legend()
    # plt.xticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
    plt.xlabel("Epoch")
    plt.savefig("plot/" + id + "Loss - Train.png")
    plt.clf()
    plt.cla()
    plt.plot(range(num_epochs), all_val_loss, label="Loss - Valid")
    plt.legend()
    # plt.xticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
    plt.xlabel("Epoch")
    plt.savefig("plot/" + id + "Loss - Valid.png")
    plt.clf()
    plt.cla()
    plt.plot(range(num_epochs), mmm, label="NDCG - Train")
    plt.legend()
    # plt.xticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
    plt.xlabel("Epoch")
    plt.savefig("plot/" + id + "NDCG - Train.png")
    plt.clf()
    plt.cla()
    plt.plot(range(num_epochs), gmmm, label="Georisk - Train")
    plt.legend()
    # plt.xticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
    plt.xlabel("Epoch")
    plt.savefig("plot/" + id + "Georisk - Train.png")
    plt.clf()
    plt.cla()
    plt.plot(range(num_epochs), mmmVal, label="NDCG - Valid")
    plt.legend()
    # plt.xticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
    plt.xlabel("Epoch")
    plt.savefig("plot/" + id + "NDCG - Valid.png")
    plt.clf()
    plt.cla()
    plt.plot(range(num_epochs), gmmmVal, label="Georisk - Valid")
    plt.legend()
    # plt.xticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
    plt.xlabel("Epoch")
    plt.savefig("plot/" + id + "Georisk - Valid.png")

    plt.clf()
    plt.cla()
