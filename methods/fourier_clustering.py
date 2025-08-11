import torch

def pca_torch(X, n_components=2):
    X_mean = torch.mean(X, dim=0)
    X = X - X_mean
    U, S, V = torch.svd(X)
    return torch.mm(X, V[:, :n_components])


def kmeans_torch(X, centroids, n_iters=10):
    n_clusters = centroids.shape[0]

    for _ in range(n_iters):
        distances = torch.cdist(X, centroids)
        labels = torch.argmin(distances, axis=1)

        for i in range(n_clusters):
            if (labels == i).any():
                centroids[i] = torch.mean(X[labels == i], axis=0)

    return labels, centroids


def process_batch(batch):
    N, C, H, W = batch.shape

    fft_results = torch.zeros(N, C, H, W, 2, device=batch.device)

    for channel in range(C):
        channel_data = batch[:, channel, :, :]
        fft_result = torch.fft.fftshift(torch.fft.fft2(channel_data), dim=(-2, -1)).abs()
        fft_results[:, channel, :, :, 0] = fft_result

    center_h_low, center_h_high = H // 4, 3 * H // 4
    center_w_low, center_w_high = W // 4, 3 * W // 4


    high_freq_data = torch.zeros_like(fft_results)

    high_freq_data[:, :, :center_h_low, :, 0] = fft_results[:, :, :center_h_low, :, 0]
    high_freq_data[:, :, center_h_high:, :, 0] = fft_results[:, :, center_h_high:, :, 0]
    high_freq_data[:, :, center_h_low:center_h_high, :center_w_low, 0] = fft_results[:, :, center_h_low:center_h_high, :center_w_low, 0]
    high_freq_data[:, :, center_h_low:center_h_high, center_w_high:, 0] = fft_results[:, :, center_h_low:center_h_high, center_w_high:, 0]

    num_samples = batch.shape[0]
    num_features = torch.prod(torch.tensor(high_freq_data.shape[1:])).item()
    reshaped_features = high_freq_data.reshape(num_samples, num_features)

    return reshaped_features

