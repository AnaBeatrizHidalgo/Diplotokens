Nós: DISEASE, SYMPTOM, DRUG, EXAM, ANATOMY, EFFECT, RESULT, PATIENT.

Arestas:
FAMILY_HISTORY, PERSONAL_HISTORY e DIAGNOSED_WITH (paciente → doença)
HAS_SYMPTOM (paciente → sintoma)
OCCURS_IN (sintoma → anatomia)
AFFECTS (doença → anatomia)
TAKES (paciente → droga)
UNDERGOES (paciente → exame)
REVEALS (exame → resultado)
HAS_EFFECT (drug → effect)