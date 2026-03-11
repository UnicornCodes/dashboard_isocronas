@echo off
title Dashboard Accesibilidad IMSS-Bienestar
cd /d "%~dp0"

echo.
echo  ============================================
echo   Accesibilidad a Centros de Salud
echo   IMSS-Bienestar - DSIS
echo  ============================================
echo.
echo  Activando entorno DataAnalytics...
call C:\Users\rosa.carbajal\AppData\Local\anaconda3\Scripts\activate.bat DataAnalytics

echo  Iniciando app en http://localhost:8050
echo  Presiona Ctrl+C para detener.
echo.
start http://localhost:8050
python app_dash.py

pause
