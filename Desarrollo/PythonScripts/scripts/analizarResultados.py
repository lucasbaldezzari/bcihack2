import os
import pandas as pd

def analizar_resultados(ruta_base):
    # Diccionario para almacenar los DataFrames de resultados
    dfs_resultados = {}
    
    # Navegar a la carpeta ClasificacionesEntrenamiento
    ruta_base = os.path.join(ruta_base, 'ClasificacionesEntrenamiento')
    
    # Iterar sobre todas las subcarpetas S1, S2, ..., Sn
    for sesion in os.listdir(ruta_base):
        ruta_sesion = os.path.join(ruta_base, sesion)
        
        # Iterar sobre las carpetas Imaginado y Movimiento
        for condicion in ['Imaginado', 'Movimiento']:
            ruta_condicion = os.path.join(ruta_sesion, condicion)
            
            # Verificar si la ruta existe antes de intentar iterar sobre las subcarpetas
            if os.path.exists(ruta_condicion):
                # Iterar sobre todas las subcarpetas presentes (MD, MI, etc.)
                for subcarpeta in os.listdir(ruta_condicion):
                    ruta_subcarpeta = os.path.join(ruta_condicion, subcarpeta)
                    
                    # Inicializar el DataFrame para esta subcarpeta si no existe
                    if subcarpeta not in dfs_resultados:
                        dfs_resultados[subcarpeta] = pd.DataFrame(columns=['Sujeto', 'LDA', 'SVM', 'CNN'])
                    
                    # Diccionario para almacenar los resultados de esta sesión
                    resultados_sesion = {'Sujeto': sesion, 'LDA': None, 'SVM': None, 'CNN': None}
                    
                    # Verificar si existe el archivo results_df.csv
                    ruta_archivo = os.path.join(ruta_subcarpeta, 'results_df.csv')
                    
                    # Utilizar try/except para manejar posibles errores de archivo no encontrado
                    try:
                        # Leer el archivo .csv
                        df = pd.read_csv(ruta_archivo)
                        
                        # Almacenar los valores de interés en el diccionario
                        resultados_sesion['LDA'] = df.iloc[0, 1]
                        resultados_sesion['SVM'] = df.iloc[0, 2]
                        resultados_sesion['CNN'] = df.iloc[0, 3]
                        
                        # Añadir los resultados de esta sesión al DataFrame de la subcarpeta correspondiente
                        dfs_resultados[subcarpeta] = dfs_resultados[subcarpeta].append(resultados_sesion, ignore_index=True)
                    except FileNotFoundError:
                        pass  # Silenciosamente ignorar si el archivo no se encuentra
                    except pd.errors.EmptyDataError:
                        pass  # Silenciosamente ignorar si el archivo está vacío
                    except Exception as e:
                        pass  # Silenciosamente ignorar otros errores
    
    return dfs_resultados

df = analizar_resultados('C:/Users/Admin/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts')
print(df)