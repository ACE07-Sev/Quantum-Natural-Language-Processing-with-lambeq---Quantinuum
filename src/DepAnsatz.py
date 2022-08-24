from __future__ import annotations

from lambeq import IQPAnsatz

from DepCircuit import(Sim13ansatz as Sim13, Sim14ansatz as Sim14, Sim15ansatz as Sim15)

"""
Circuit Ansatz
==============
A circuit ansatz is used to convert a DisCoCat diagram into a quantum circuit.

"""

__all__ = ['CircuitAnsatz', 'IQPAnsatz']

from collections.abc import Mapping
from typing import Any, Callable, Optional

from discopy.quantum.circuit import (Circuit, Discard, Functor, Id, qubit)
from discopy.quantum.gates import Bra, Ket, Rx, Rz
from discopy.rigid import Box, Diagram, Ty
import numpy as np

from lambeq.ansatz import BaseAnsatz, Symbol

_ArMapT = Callable[[Box], Circuit]


class CircuitAnsatz(BaseAnsatz):
    """Base class for circuit ansatz."""

    def __init__(self, ob_map: Mapping[Ty, int], **kwargs: Any) -> None:
        """Instantiate a circuit ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the number of
            qubits it uses in a circuit.
        **kwargs : dict
            Extra parameters for ansatz configuration.

        """
        self.ob_map = ob_map
        self.functor = Functor({}, {})

    def __call__(self, diagram: Diagram) -> Circuit:
        """Convert a DisCoPy diagram into a DisCoPy circuit."""
        return self.functor(diagram)

    def _ob(self, pg_type: Ty) -> int:
        """Calculate the number of qubits used for a given type."""
        return sum(self.ob_map[Ty(factor.name)] for factor in pg_type)

    def _special_cases(self, ar_map: _ArMapT) -> _ArMapT:
        """Convert a DisCoPy box into a tket Circuit element"""
        return ar_map

def _sim_ansatz_factory(sim_circuit_n=13):
    """Generate a SimAnsatz class for the specified Sim ansatz.

    Parameters
    ----------
    sim_circuit_n : int
        Circuit number, according to arXiv:1905.10876

    """

    # Map from Sim circuit number to a function specifying
    # number of params, given the number of qubits
    param_count_fns = {13: lambda n_qbs: 4 * n_qbs,
                       14: lambda n_qbs: 4 * n_qbs,
                       15: lambda n_qbs: 2 * n_qbs}

    sim_discopy_classes = {13: Sim13, 14: Sim14, 15: Sim15}

    if sim_circuit_n not in sim_discopy_classes:
        raise ValueError(f'Invalid Sim circuit name: {sim_circuit_n}')

    class SimAnsatz(CircuitAnsatz):
        """ Sim circuit ansatz (arXiv:1905.10876)

        Multiple Sim ansatze can be created, using the available
         DisCoPy classes specified in `sim_discopy_classes`.

        """

        def __init__(self,
                     ob_map: Mapping[Ty, int],
                     n_layers: int,
                     n_single_qubit_params: int = 3,
                     discard: bool = False,
                     special_cases: Optional[Callable[[_ArMapT],
                                                      _ArMapT]] = None):
            """Instantiate Sim circuit ansatz (arXiv:1905.10876).

            Parameters
            ----------
            ob_map : dict
                A mapping from :py:class:`discopy.rigid.Ty` to the number of
                qubits it uses in a circuit.
            n_layers : int
                The number of IQP layers used by the ansatz.
            n_single_qubit_params : int, default: 3
                The number of single qubit rotations used by the ansatz.
            discard : bool, default: False
                Discard open wires instead of post-selecting.
            special_cases : callable, optional
                A function that transforms an arrow map into one specifying
                special cases that should not be converted by the Ansatz
                class.

            """

            super().__init__(ob_map=ob_map, n_layers=n_layers,
                             n_single_qubit_params=n_single_qubit_params)

            if special_cases is None:
                special_cases = self._special_cases

            self.n_layers = n_layers
            self.n_single_qubit_params = n_single_qubit_params
            self.discard = discard
            self.functor = Functor(ob=self.ob_map,
                                   ar=special_cases(self._ar))

        def _ar(self, box: Box) -> Circuit:
            
            # Step 1: obtain labels
            label = self._summarise_box(box)
            
            #Step 2: Apply functor to the domain and codomain
            dom, cod = self._ob(box.dom), self._ob(box.cod)

            n_qubits = max(dom, cod)
            n_layers = self.n_layers
            n_1qubit_params = self.n_single_qubit_params

             #Step 3: Construct and return ansatz with new domain and codomain
            if n_qubits == 0:
                circuit = Id()
            elif n_qubits == 1:
                syms = [Symbol(f'{label}_{i}') for i in range(n_1qubit_params)]
                rots = [Rx, Rz]
                circuit = Id(qubit)
                for i, sym in enumerate(syms):
                    circuit >>= rots[i % 2](sym)
            else:
                n_params = n_layers * param_count_fns[sim_circuit_n](n_qubits)
                syms = [Symbol(f'{label}_{i}') for i in range(n_params)]
                params = np.array(syms).reshape(
                        (n_layers, param_count_fns[sim_circuit_n](n_qubits)))

                circuit = sim_discopy_classes[sim_circuit_n](n_qubits, params)

            if cod > dom:
                circuit <<= Id(dom) @ Ket(*[0]*(cod - dom))
            elif self.discard:
                circuit >>= Id(cod) @ Discard(dom - cod)
            else:
                circuit >>= Id(cod) @ Bra(*[0]*(dom - cod))
            return circuit

    return SimAnsatz

Sim13Ansatz = _sim_ansatz_factory(13)
Sim14Ansatz = _sim_ansatz_factory(14)
Sim15Ansatz = _sim_ansatz_factory(15)
