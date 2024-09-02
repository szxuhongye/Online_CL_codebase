            with torch.no_grad():
                self.net.eval()
                out, cout = self.net(x)
                out = out.data.cpu().numpy()[0]
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()[0]
                maxInds = np.argmax(batchProbs)
                embDim = len(out)
                embedding = np.zeros(embDim * num_classes)
                               
                order = np.argsort(batchProbs)[::-1]
                probs = batchProbs[order]
                for c in range(num_classes):
                    if c == maxInds:
                        embedding[embDim * c : embDim * (c+1)] = deepcopy(out) * (1 - probs[c])
                    else:
                        embedding[embDim * c : embDim * (c+1)] = deepcopy(out) * (-1 * probs[c])
               
                probs = probs / np.sum(probs)

                embedding = embedding * np.sqrt(probs)
            return torch.tensor(embedding, device=self.device).float()
    

# Next, here is the code to obtain Bayesian distributions of the predictions. self.sigma is a matrix from vessal paper. I calculate it like this, so it serves as fisher information matrix:

# UPDATE SIGMA WITH NEW GRADS
        sigma2 = self.sigma @ torch.outer(grads, grads) @ self.sigma
        sigma3 = sigma2/ (1 + grads @ self.sigma @ grads)
        self.sigma = (self.sigma - sigma3)
            



# This is the main code:
            emb, logits =  self.net(self.batch[0])
            preds = F.softmax(logits, dim=1)
            exoponential = torch.exp(logits)
            sums = exoponential.sum(dim=1)
            softmax_derivative = ((exoponential * sums.unsqueeze(1)) - exoponential **2) * (1/sums.unsqueeze(1))**2
            J = torch.einsum('bi,bj->bij', (softmax_derivative, emb))
            JJ = torch.zeros((emb.shape[0], self.num_classes, self.num_classes*emb.shape[1])).cuda()
            for i in range(self.num_classes):
                JJ[:,i,i*emb.shape[1]:(i+1)*emb.shape[1]] = J[:,i,:]
           
            # Prrediction dist (Normal distribution over weights -> Normal distribution over outputs)
            sigma = torch.einsum('bcn,nn,bnl->bcl', (JJ, self.sigma, JJ.transpose(1,2)))
           
            # Laplace bridge  (Normal distribution over outputs -> Dirichlet distribution)
            _ , K = preds.size(0), preds.size(-1)
            sigma_diagonal = torch.diagonal(sigma, dim1=1, dim2=2).cpu()
            sum_exp = torch.sum(torch.exp(-1*preds), dim=1).view(-1,1)
            alphas = 1/sigma_diagonal * (1 - 2/K + torch.exp(preds)/K**2 * sum_exp)
           
            # Entropy of Dirichlet distribution
            dist = torch.distributions.Dirichlet(alphas)