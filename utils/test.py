from utils.utils import progress_bar


def test(model, criterion, testloader, device):
    model.eval()
    print('---------------------------------------------Test------------------------------------------')
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(testloader):
        data, target = data.to(device), target.to(device)
        output = model(data, device)
        #test_loss += criterion(output, target).data.item()
        #test_loss += criterion(output, target).data.item()
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        # 保持维度不知道是为了什么，但保持好吧；.max()或者.min()返回的是一个二元组，(最大值，索引)
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()  #不加cpu()行吗
        correct += pred.eq(target.data.view_as(pred)).sum()

        progress_bar(batch_idx, len(testloader), msg='test')

    test_loss /= len(testloader.dataset)
    '''固定用法，不用做什么改变就可以用；len(testloader)，len(testloader.dataset)'''

    print('\nTestset: Average loss: %.4f, Accuracy: %d/%d (%%%.2f)\n'
            %(test_loss, correct, len(testloader.dataset), 100.0 * float(correct) / len(testloader.dataset)))
    return (test_loss, float(correct) / len(testloader.dataset))