W repozytorium `FEniCS` znajduje się: 

- plik `unsteady_Stokes_diploma_wariantA.py`
    - plik ten zawiera kod, zapisany za pomocą Pythonowej składni realizujący warunek (3.29) z pracy dyplomowej. 
    
- plik `unsteady_Stokes_diploma_wariantB.py`
    - plik ten zawiera kod, zapisany za pomocą Pythonowej składni realizujący warunek (3.30) z pracy dyplomowej. 
    
Aby uruchomić powyższe pliki, konieczne jest zainstalowanie biblioteki FEniCS. Instrukcja: https:[//fenicsproject.org/download/]

Konieczne jest również zainstalowanie bibliotek: `numpy, petsc4py` oraz `tqdm`.

**UWAGA NA PROBLEMY ZE ZGODNOŚCIĄ!**  
Powyższy kod tworzony był na platformie docker. W celu uruchomienia kodu zaleca się zainstalowanie biblioteki FEniCS w wersji 2018.1.0 i korzystanie z niej przez tę platformę.


- plik wyplywy_wizualizacja.py 
    - plik ten zawiera kod zapisany za pomocą Pythonowej składni realizujący wykresy:
        1. wykres 5.1 z pracy dyplomowej. 
        2. wykres 5.6 z pracy dyplomowej. 
    
Aby uruchomić powyższy plik, konieczne jest zainstalowaie bibliotek `matplotlib.pyplot`, `pandas` oraz `numpy`. 

Kompilacja nastąpi, jeżeli w folderze roboczym znajdować się będą foldery z wynikami symulacji przepływowych. Te są autoamtycznie tworzone po przekompilowaniu plików `unsteady_stokes_diploma_wariantA.py` oraz `unsteady_Stokes_diploma_wariantB.py`. W pliku należy ręcznie ustawić, jakie dane chcemy porównywać. 

- folder siatki
Folder zawiera podfoldery siatek stworzonych za pomocą programu GMSH (w folderze siatki_gmsh) oraz przekonwertowane na format akceptowany dla platformy FEniCS (w folderze siatki_xml) 
    
Aby móc otworzyć pliki .msh należy zainstalować program Gmsh. Więcej informacji na stronie: [http://gmsh.info]. Pliki `.xml` są otwierane z poziomu FEniCSa. 


![](T_pressure.avi)
