# PyTorch dependencies
import torch
import torch.nn.functional as F
from torch.autograd import Variable

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM_Trainer():
    def __init__(self):
        self.embedding_dimension = 200
        self.hidden_dimension = 200
    
    def train(self, model, train_loader, validation_loader, epochs, lr):
        train_loss_graph = []
        val_loss_graph = []
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr)

        model.to(default_device)
        #print(len(train_loader))
        print("\n\n")

        print("Training the network:\n-------------------------------------------------------------\n")
        print("Device:", default_device)
        print("\nLSTM Model Architecture:\n", model)
        print("\n")
        for epoch in range(1, epochs + 1):
            print("Epoch", epoch, "\n")
            model.train()
            sum_loss = 0.0
            total = 0

            for batch_idx, (x, y) in enumerate(train_loader, 0):

                x = x.long()
                y = y.long()

                if torch.cuda.is_available():
                    x = Variable(x.cuda(default_device))
                    y = Variable(y.cuda(default_device))

                y_pred,_ = model(x)

                optimizer.zero_grad()
                loss = F.cross_entropy(y_pred, y)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]

                if batch_idx%200 == 0:
                    print("Batch", batch_idx, "/", len(train_loader), ". Loss: ", str(loss.item()))

            val_loss, val_acc = self.validation(model, validation_loader)
            train_loss_graph.append(sum_loss/total)
            val_loss_graph.append(val_loss)


            print("Train Loss: %.5f, Validation Loss: %.5f, Validation Accuracy: %.5f" % (sum_loss/total, val_loss, val_acc))
            print("Epoch end\n------------------------------------------------\n\n\n")


        return train_loss_graph, val_loss_graph

    def validation(self, model, validation_loader):

        model.eval()
        correct = 0
        total = 0
        sum_loss = 0.0

        for x, y in validation_loader:

            x = x.long()
            y = y.long()

            if torch.cuda.is_available():
                x = Variable(x.cuda(default_device))
                y = Variable(y.cuda(default_device))

            y_hat,_ = model(x)

            loss = F.cross_entropy(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item() * y.shape[0]


        return sum_loss / total, correct / total


    def test(self, model, validation_loader):

        model.eval()
        targets=[]
        predictions=[]
        correct = 0
        total = 0
        sum_loss = 0.0
        #testset_final_feature_vectors = []

        print("Testing the network:\n-------------------------------------------------------------\n")
        for x, y in validation_loader:

            x = x.long()
            y = y.long()

            if torch.cuda.is_available():
                x = Variable(x.cuda(default_device))
                y = Variable(y.cuda(default_device))

            output, feature_vector = model(x)
            loss = F.cross_entropy(output, y)

            _, pred = torch.max(output, 1)
            correct += (pred == y).float().sum()

            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]

            targets.extend(y.cpu().tolist())
            predictions.extend(pred.cpu().tolist())

            """
            #TODO: New Code - Experimental!
            feature_vector = feature_vector.tolist()
            testset_final_feature_vectors.append(feature_vector)

        import itertools
        testset_final_feature_vectors = list(itertools.chain(*testset_final_feature_vectors))

        #print(len(testset_final_feature_vectors))

        with open('testset_feature_vectors.txt', 'w') as f:
            for item in testset_final_feature_vectors:
                f.write("{}\n".format(item))

        from sklearn.metrics.pairwise import cosine_similarity
        cos_matrix = cosine_similarity(testset_final_feature_vectors, dense_output = False)
        #print(len(cos_matrix))
        #print(len(cos_matrix[0]))

        with open('cosine_similarity_matrix.txt', 'w') as f:
            for item in cos_matrix:
                f.write("{}\n".format(item))
            """

        test_acc = correct/total

        print("Test accuracy:", test_acc.item())

        return targets, predictions, test_acc