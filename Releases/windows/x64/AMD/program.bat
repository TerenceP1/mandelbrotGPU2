@echo off
echo This is incomplete and is in testing. Do not use.
cd internal
zluda.exe DeviceSelecter.exe
for /f %%i in ('python -c print("pythonpy",end="")') do (
  SET x=%%i
)
if (%x == "pythonpy") (
  echo python detected!
)
else (
  echo no phthon found
)
