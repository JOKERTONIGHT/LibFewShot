from .metric_model import MetricModel
from abc import abstractmethod
import torch
from torch import Tensor, nn
from typing import Optional
from collections import OrderedDict

# utils
def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """

    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )

# def entropy(logits: Tensor) -> Tensor:
#     """
#     Compute entropy of prediction.
#     WARNING: takes logit as input, not probability.
#     Args:
#         logits: shape (n_images, n_way)
#     Returns:
#         Tensor: shape(), Mean entropy.
#     """
#     probabilities = logits.softmax(dim=1)
#     return (-(probabilities * (probabilities + 1e-12).log()).sum(dim=1)).mean()

# def k_nearest_neighbours(features: Tensor, k: int, p_norm: int = 2) -> Tensor:
#     """
#     Compute k nearest neighbours of each feature vector, not included itself.
#     Args:
#         features: input features of shape (n_features, feature_dimension)
#         k: number of nearest neighbours to retain
#         p_norm: use l_p distance. Defaults: 2.

#     Returns:
#         Tensor: shape (n_features, k), indices of k nearest neighbours of each feature vector.
#     """
#     distances = torch.cdist(features, features, p_norm)

#     return distances.topk(k, largest=False).indices[:, 1:]

# def power_transform(features: Tensor, power_factor: float) -> Tensor:
#     """
#     Apply power transform to features.
#     Args:
#         features: input features of shape (n_features, feature_dimension)
#         power_factor: power to apply to features

#     Returns:
#         Tensor: shape (n_features, feature_dimension), power transformed features.
#     """
#     return (features.relu() + 1e-6).pow(power_factor)

# def strip_prefix(state_dict: OrderedDict, prefix: str):
#     """
#     Strip a prefix from the keys of a state_dict. Can be used to address compatibility issues from
#     a loaded state_dict to a model with slightly different parameter names.
#     Example usage:
#         state_dict = torch.load("model.pth")
#         # state_dict contains keys like "module.encoder.0.weight" but the model expects keys like "encoder.0.weight"
#         state_dict = strip_prefix(state_dict, "module.")
#         model.load_state_dict(state_dict)
#     Args:
#         state_dict: pytorch state_dict, as returned by model.state_dict() or loaded via torch.load()
#             Keys are the names of the parameters and values are the parameter tensors.
#         prefix: prefix to strip from the keys of the state_dict. Usually ends with a dot.

#     Returns:
#         copy of the state_dict with the prefix stripped from the keys
#     """
#     return OrderedDict(
#         [
#             (k[len(prefix) :] if k.startswith(prefix) else k, v)
#             for k, v in state_dict.items()
#         ]
#     )

class FewShotClassifier(MetricModel):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        use_softmax: bool = False,
        feature_centering: Optional[Tensor] = None,
        feature_normalization: Optional[float] = None,
    ):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: the feature extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method).
                If None is passed, the backbone will be initialized as nn.Identity().
            use_softmax: whether to return predictions as soft probabilities
            feature_centering: a features vector on which to center all computed features.
                If None is passed, no centering is performed.
            feature_normalization: a value by which to normalize all computed features after centering.
                It is used as the p argument in torch.nn.functional.normalize().
                If None is passed, no normalization is performed.
        """
        super().__init__()

        self.backbone = backbone if backbone is not None else nn.Identity()
        self.use_softmax = use_softmax

        self.prototypes = torch.tensor(())
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())

        self.feature_centering = (
            feature_centering if feature_centering is not None else torch.tensor(0)
        )
        self.feature_normalization = feature_normalization

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "All few-shot algorithms based on this approach must implement a set_forward method."
        )
    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        raise NotImplementedError(
            "All few-shot algorithms based on this approach must implement a set_forward_loss method."
        )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Harness information from the support set, so that query labels can later be predicted using a forward call.
        The default behaviour shared by most few-shot classifiers is to compute prototypes and store the support set.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    @staticmethod
    def is_transductive() -> bool:
        raise NotImplementedError(
            "All few-shot algorithms must implement a is_transductive method."
        )

    def compute_features(self, images: Tensor) -> Tensor:
        """
        Compute features from images and perform centering and normalization.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension)
        """
        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(
                centered_features, p=self.feature_normalization, dim=1
            )
        return centered_features

    def softmax_if_specified(self, output: Tensor, temperature: float = 1.0) -> Tensor:
        """
        If the option is chosen when the classifier is initialized, we perform a softmax on the
        output in order to return soft probabilities.
        Args:
            output: output of the forward method of shape (n_query, n_classes)
            temperature: temperature of the softmax
        Returns:
            output as it was, or output as soft probabilities, of shape (n_query, n_classes)
        """
        return (temperature * output).softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:
        """
        Compute prediction logits from their euclidean distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) -> Tensor:
        """
        Compute prediction logits from their cosine distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return (
            nn.functional.normalize(samples, dim=1)
            @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def compute_prototypes_and_store_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Extract support features, compute prototypes, and store support labels, features, and prototypes.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.support_labels = support_labels
        self.support_features = self.compute_features(support_images)
        self._raise_error_if_features_are_multi_dimensional(self.support_features)
        self.prototypes = compute_prototypes(self.support_features, support_labels)

    @staticmethod
    def _raise_error_if_features_are_multi_dimensional(features: Tensor):
        if len(features.shape) != 2:
            raise ValueError(
                "Illegal backbone or feature shape. "
                "Expected output for an image is a 1-dim tensor."
            )

class BDCSPN(FewShotClassifier):
    
    """
    Jinlu Liu, Liang Song, Yongqiang Qin
    "Prototype Rectification for Few-Shot Learning" (ECCV 2020)
    https://arxiv.org/abs/1911.10713

    Rectify prototypes with label propagation and feature shifting.
    Classify queries based on their cosine distance to prototypes.
    This is a transductive method.
    """
    
    def rectify_prototypes(
        self, query_features: Tensor
    ):  # pylint: disable=not-callable
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features of shape (n_query, feature_dimension)
        """
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = nn.functional.one_hot(self.support_labels, n_classes)

        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift

        support_logits = self.cosine_distance_to_prototypes(self.support_features).exp()
        query_logits = self.cosine_distance_to_prototypes(query_features).exp()

        one_hot_query_prediction = nn.functional.one_hot(
            query_logits.argmax(-1), n_classes
        )

        normalization_vector = (
            (one_hot_support_labels * support_logits).sum(0)
            + (one_hot_query_prediction * query_logits).sum(0)
        ).unsqueeze(
            0
        )  # [1, n_classes]
        support_reweighting = (
            one_hot_support_labels * support_logits
        ) / normalization_vector  # [n_support, n_classes]
        query_reweighting = (
            one_hot_query_prediction * query_logits
        ) / normalization_vector  # [n_query, n_classes]

        self.prototypes = (support_reweighting * one_hot_support_labels).t().matmul(
            self.support_features
        ) + (query_reweighting * one_hot_query_prediction).t().matmul(query_features)

    # def forward(
    #     self,
    #     query_images: Tensor,
    # ) -> Tensor:
    #     """
    #     Overrides forward method of FewShotClassifier.
    #     Update prototypes using query images, then classify query images based
    #     on their cosine distance to updated prototypes.
    #     """
    #     query_features = self.compute_features(query_images)

    #     self.rectify_prototypes(
    #         query_features=query_features,
    #     )
    #     return self.softmax_if_specified(
    #         self.cosine_distance_to_prototypes(query_features)
    #     )
    
    def set_forward(self, query_images: Tensor) -> Tensor:
        """
        推理阶段的前向传播逻辑。
        """
        query_features = self.compute_features(query_images)
        self.rectify_prototypes(query_features=query_features)
        distances = self.cosine_distance_to_prototypes(query_features)
        acc = (distances.argmax(1) == self.support_labels[None]).float
        return self.softmax_if_specified(distances), acc

    def set_forward_loss(self, query_images: Tensor, query_labels: Tensor) -> Tensor:
        """
        训练阶段的损失计算逻辑。
        """
        predictions, acc = self.set_forward(query_images)
        loss = self.loss_function(predictions, query_labels)
        return predictions, acc, loss

    @staticmethod
    def is_transductive() -> bool:
        return True