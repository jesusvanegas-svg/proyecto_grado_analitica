import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard | Lead Time vs Paradas", layout="wide", page_icon="🏭")

st.markdown("""
    <style>
    .titulo { font-size: 28px; font-weight: bold; color: #0F172A; }
    .subtitulo { font-size: 16px; color: #475569; margin-bottom: 20px;}
    .alerta-roja { background-color: #FEE2E2; color: #991B1B; padding: 15px; border-radius: 8px; font-weight: bold; }
    .alerta-verde { background-color: #D1FAE5; color: #065F46; padding: 15px; border-radius: 8px; font-weight: bold; }
    .metric-card { background-color: #F8FAFC; border-radius: 8px; padding: 15px; border-left: 5px solid #2563EB; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="titulo">Dashboard Analítico: Lead Time, Paradas y Costos</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Análisis predictivo y financiero cruzado por Orden de Producción</div>', unsafe_allow_html=True)

# ==========================================
# 2. LECTURA AUTOMÁTICA DE ARCHIVOS
# ==========================================
# El sistema leerá estos archivos directamente de tu GitHub
file_lead = "LEAD TIME DE PLANTA 2023-2026.xlsx"
file_paradas = "PARADAS DE MAQUINA 2024-2026.xlsx"
file_costos = "Costeo Reprocesos FL.xlsx"

try:
    with st.spinner('Procesando millones de datos en la nube...'):
        # --- LECTURA INICIAL ---
        df_lead = pd.read_excel(file_lead)
        df_paradas = pd.read_excel(file_paradas, header=1) # Títulos en la fila 1
        
        df_lead.columns = df_lead.columns.str.strip()
        df_paradas.columns = df_paradas.columns.str.strip()

        # ==========================================
        # 3. ETL Y CRUCE DE BASES (Lead Time + Paradas)
        # ==========================================
        df_paradas = df_paradas.dropna(subset=['Orden']) 
        
        df_paradas_agg = df_paradas.groupby('Orden').agg(
            maquinas_parada=('Maquina', lambda x: ', '.join(x.dropna().astype(str).unique())),
            causa_principal=('Paradas', lambda x: x.mode()[0] if not x.empty else 'Desconocida'),
            minutos_totales=('Minutos', 'sum')
        ).reset_index()

        mask_acabados = df_paradas['Proceso'].astype(str).str.contains('ACABADOS|AC', case=False, na=False) | \
                        df_paradas['Maquina'].astype(str).str.contains('ACABADOS', case=False, na=False)
        df_acabados = df_paradas[mask_acabados]
        df_acabados_agg = df_acabados.groupby('Orden')['Minutos'].sum().reset_index()
        df_acabados_agg.rename(columns={'Minutos': 'minutos_acabados'}, inplace=True)

        df_master = pd.merge(df_lead, df_paradas_agg, on='Orden', how='inner')
        df_master = pd.merge(df_master, df_acabados_agg, on='Orden', how='left')
        df_master['minutos_acabados'] = df_master['minutos_acabados'].fillna(0)
        
        df_master['cumple_meta'] = np.where(df_master['Tiempo total en dias'] <= 17, 'Cumple (<= 17 días)', 'Incumple (> 17 días)')
        df_master['impacto_acabados_dias'] = df_master['minutos_acabados'] / 1440
        df_master['reproceso_bin'] = np.where(df_master['Reprocesada'].astype(str).str.upper() == 'SI', 1, 0)

        # ==========================================
        # 4. ETL FINANCIERO (Solución del Error)
        # ==========================================
        df_costos_raw = pd.read_excel(file_costos)
        df_maquinas_costo = df_costos_raw.iloc[8:17].copy()
        
        # Renombramos las columnas usando su posición para evitar errores de espacios ocultos
        df_maquinas_costo.rename(columns={
            df_maquinas_costo.columns[0]: 'maquina',
            df_maquinas_costo.columns[8]: 'costo_total',
            df_maquinas_costo.columns[10]: 'costo_minuto'
        }, inplace=True)
        
        df_maquinas_costo = df_maquinas_costo.dropna(subset=['maquina']).reset_index(drop=True)
        df_maquinas_costo['costo_total'] = pd.to_numeric(df_maquinas_costo['costo_total'], errors='coerce').fillna(0)
        df_maquinas_costo['costo_minuto'] = pd.to_numeric(df_maquinas_costo['costo_minuto'], errors='coerce').fillna(0)

        # MENÚ LATERAL AUTOMÁTICO
        st.sidebar.success("✅ Conectado a la Base de Datos")
        st.sidebar.info("Este dashboard lee automáticamente la información de la planta validada para el periodo 2023-2026 directamente desde la nube.")
        st.sidebar.markdown("---")
        st.sidebar.metric(label="Órdenes Analizadas", value=f"{len(df_master):,}")

        # ==========================================
        # 5. DASHBOARD - VISUALIZACIONES
        # ==========================================
        tabs = st.tabs(["📊 Referencias y Reprocesos", "⚠️ Causas y Máquinas", "🧮 Modelo Predictivo", "💰 Impacto Financiero"])

        with tabs[0]: 
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("1. Estado de Referencias (>17 días)")
                top_refs = df_master['Referencia Color'].value_counts().head(10).index
                df_refs_top = df_master[df_master['Referencia Color'].isin(top_refs)]
                df_refs_plot = df_refs_top.groupby(['Referencia Color', 'cumple_meta']).size().reset_index(name='cantidad')
                fig_refs = px.bar(df_refs_plot, x='Referencia Color', y='cantidad', color='cumple_meta', barmode='group',
                                  color_discrete_map={'Cumple (<= 17 días)': '#10B981', 'Incumple (> 17 días)': '#EF4444'}, title="Top 10 Referencias")
                st.plotly_chart(fig_refs, use_container_width=True)

            with col2:
                st.subheader("2. Impacto del Reproceso")
                df_rep = df_master.groupby('Reprocesada').agg(promedio_dias=('Tiempo total en dias', 'mean'), promedio_minutos=('minutos_totales', 'mean')).reset_index()
                fig_rep = go.Figure()
                fig_rep.add_trace(go.Bar(x=df_rep['Reprocesada'].astype(str), y=df_rep['promedio_dias'], name='Lead Time (Días)', marker_color='#3B82F6', yaxis='y1'))
                fig_rep.add_trace(go.Scatter(x=df_rep['Reprocesada'].astype(str), y=df_rep['promedio_minutos'], name='Paradas (Min)', mode='lines+markers', line=dict(color='#F59E0B', width=3), yaxis='y2'))
                fig_rep.update_layout(title="Doble Castigo: Días vs Minutos", yaxis=dict(title="Lead Time Promedio"), yaxis2=dict(title="Minutos de Parada", overlaying='y', side='right'))
                st.plotly_chart(fig_rep, use_container_width=True)

        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("3. Causas Principales de Parada")
                df_incumplen = df_master[df_master['cumple_meta'] == 'Incumple (> 17 días)']
                df_causas = df_incumplen.groupby('causa_principal')['minutos_totales'].sum().reset_index().sort_values('minutos_totales', ascending=False).head(8)
                fig_causas = px.bar(df_causas, x='minutos_totales', y='causa_principal', orientation='h', color='minutos_totales', color_continuous_scale='Reds', title="Minutos perdidos por causa")
                st.plotly_chart(fig_causas, use_container_width=True)
                
            with c2:
                st.subheader("4. Máquinas: Teórica vs Real")
                st.dataframe(df_master[['Orden', 'Maquina trabaajo', 'maquinas_parada', 'causa_principal', 'minutos_totales']].head(10), use_container_width=True)
                st.info("💡 **Hallazgo de Acabados:** Las paradas en la sección de Acabados se registran en minutos, pero penalizan la entrega transformándose en días completos de retraso.")

        with tabs[2]:
            st.subheader("5. Motor Predictivo (Machine Learning)")
            df_modelo = df_master.dropna(subset=['minutos_totales', 'minutos_acabados', 'reproceso_bin', 'Metros a Fabri', 'Tiempo total en dias'])
            X = df_modelo[['Metros a Fabri', 'minutos_totales', 'impacto_acabados_dias', 'reproceso_bin']]
            y = df_modelo['Tiempo total en dias']
            modelo = LinearRegression().fit(X, y)

            col_a, col_b = st.columns(2)
            with col_a:
                metros = st.number_input("Metros a Fabricar:", min_value=100.0, value=15000.0)
                sel_rep = st.selectbox("¿Tendría Reproceso?:", ["NO", "SI"])
                val_min_parada = st.number_input("Estimación Parada General (Minutos):", min_value=0.0, value=120.0)
                val_min_acabados = st.number_input("Estimación Parada en ACABADOS (Minutos):", min_value=0.0, value=60.0)
            
            with col_b:
                input_rep = 1 if sel_rep == "SI" else 0
                input_acabados_dias = val_min_acabados / 1440
                pred = modelo.predict([[metros, val_min_parada, input_acabados_dias, input_rep]])[0]
                
                st.markdown(f"### Lead Time Proyectado: **{pred:.1f} Días**")
                if pred > 17:
                    st.markdown(f"<div class='alerta-roja'>🚨 RIESGO ALTO: La orden superará la meta de 17 días.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alerta-verde'>✅ RIESGO BAJO: La orden llegará a tiempo.</div>", unsafe_allow_html=True)
                
                st.markdown(f"**Pesos Matemáticos:** El reproceso añade **+{modelo.coef_[3]:.2f} días** | Acabados añade **+{(modelo.coef_[2] * input_acabados_dias):.2f} días**.")

        with tabs[3]:
            st.subheader("6. Análisis Financiero de Reprocesos")
            costo_total = df_maquinas_costo['costo_total'].sum()
            st.markdown(f"<div class='metric-card' style='border-left-color: #10B981;'><h2>Fuga de Capital Total: ${costo_total:,.0f}</h2></div><br>", unsafe_allow_html=True)
            
            fig_costos = px.bar(
                df_maquinas_costo.sort_values('costo_total', ascending=False), 
                x='maquina', y='costo_total', color='costo_minuto',
                title="Costo de Reproceso vs. Costo por Minuto Operativo",
                labels={'costo_total': 'Costo Acumulado ($)', 'maquina': 'Máquina', 'costo_minuto': 'Costo por Minuto ($)'}
            )
            st.plotly_chart(fig_costos, use_container_width=True)
            st.info("💡 **Conclusión Financiera:** Las Estampadoras concentran la mayor cantidad de pérdidas debido a que su minuto inactivo es el más costoso de la planta ($5,738).")

except FileNotFoundError:
    st.error("❌ Archivos no encontrados.")
    st.warning("Asegúrate de que tus 3 archivos Excel estén guardados EXACTAMENTE con estos nombres en tu repositorio de GitHub: 'LEAD TIME DE PLANTA 2023-2026_2.xlsx', 'PARADAS DE MAQUINA 2024-2026_2.xlsx' y 'Costeo Reprocesos FL.xlsx'")
