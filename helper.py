import matplotlib.pyplot as plt


def write_figures(location, train_losses, val_losses, train_accuracy, val_accuracy):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig(location + '/loss.png')
    plt.close('all')
    plt.plot(train_accuracy, label='training accuracy')
    plt.plot(val_accuracy, label='validation accuracy')
    plt.legend()
    plt.savefig(location + '/accuracy.png')
    plt.close('all')


def write_log(location, epoch, train_loss, val_loss, train_accuracy, val_accuracy):
    if epoch == 0:
        f = open(location + '/log.txt', 'w+')
        f.write('epoch\t\ttrain_loss\t\tval_loss\t\ttrain_acc\t\tval_acc\n')
    else:
        f = open(location + '/log.txt', 'a+')

    f.write(str(epoch) + '\t' + str(train_loss) + '\t' + str(val_loss) + '\t' + str(
        train_accuracy) + '\t' + str(val_accuracy) + '\n')

    f.close()