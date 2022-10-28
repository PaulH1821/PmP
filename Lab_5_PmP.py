from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD



model=BayesianNetwork(
    [
        ('Card1','Player1'),
        ('Card2','Player2'),
        ('')

    ]
    )
cpd_card=TabularCPD('card', 5 ,[[0.20],[0.20],[0.20],[0.20],[0.20]] )