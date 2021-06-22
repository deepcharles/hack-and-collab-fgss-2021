import numpy as np
from sklearn.decomposition import SparseCoder

from utils import get_dictionary_from_single_atom


class ConvSparseCoder:
    def __init__(self, atoms: np.ndarray, positive_code: bool = True) -> None:
        self.atoms = atoms
        self.n_atoms, self.atom_width = self.atoms.shape
        self.positive_code = positive_code

    def fit(self, signal: np.ndarray, penalty: float) -> "ConvSparseCoder":
        n_samples = signal.shape[0]

        self.dictionary = np.vstack(
            [
                get_dictionary_from_single_atom(atom=atom, n_samples=n_samples)
                for atom in self.atoms
            ]
        )

        # actual sparse coding
        sparse_codes_flat = ConvSparseCoder.get_sparse_codes(
            signal=signal,
            dictionary=self.dictionary,
            penalty=penalty,
            positive_code=self.positive_code,
        )
        # reshape
        self.sparse_codes = np.vstack(
            np.split(sparse_codes_flat[0], self.n_atoms)
        )
        return self

    @staticmethod
    def get_sparse_codes(
        signal: np.ndarray,
        dictionary: np.ndarray,
        penalty: float,
        positive_code: bool,
    ) -> np.ndarray:
        """Compute the sparse codes for a given signal and dictionary."""
        coder = SparseCoder(
            dictionary=dictionary,
            transform_algorithm="lasso_lars",
            transform_alpha=penalty,
            positive_code=positive_code,
        )
        return coder.transform(signal.reshape(1, -1))

    def predict(self):
        reconstruction = sum(
            np.convolve(codes, atom, mode="full")
            for (codes, atom) in zip(self.sparse_codes, self.atoms)
        )
        return reconstruction
