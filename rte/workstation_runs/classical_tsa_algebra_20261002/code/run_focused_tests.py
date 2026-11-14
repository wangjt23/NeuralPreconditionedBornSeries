"""Run the isolated TSA identity tests and retain their exact counts."""
import json
from pathlib import Path
import sys
import time
import unittest

class Counted(unittest.TextTestResult):
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.subtests=0
    def addSubTest(self,test,subtest,err):
        self.subtests+=1;super().addSubTest(test,subtest,err)

root=Path(__file__).resolve().parent
started=time.perf_counter()
suite=unittest.defaultTestLoader.discover(str(root),pattern="test_algebraic_tsa.py")
result=unittest.TextTestRunner(verbosity=2,resultclass=Counted).run(suite)
receipt=dict(tests=result.testsRun,subtests=result.subtests,passed=result.wasSuccessful(),
    failures=len(result.failures),errors=len(result.errors),seconds=time.perf_counter()-started)
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
