# -*- coding: utf-8 -*-
"""
app.py
基于蒙特卡洛模拟的滨海高桩承台桥墩地震失效模式概率评估系统 (纯 NumPy 零依赖版 + Excel导出)
"""

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import io

# ================== 1. 网页全局配置 ==================
st.set_page_config(
    page_title="Seismic Failure Mode Probability Assessment",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 95% !important; }
    div[data-testid="column"] { display: flex; flex-direction: column; justify-content: center; }
    hr { margin-top: 5px; margin-bottom: 15px; }
    div[data-baseweb="input"] input {
        text-align: center !important;
    }  
    </style>
""", unsafe_allow_html=True)

# ================== 2. 模型加载缓存 (纯 NumPy) ==================
@st.cache_resource
def load_numpy_model():
    assets_path = 'model_assets_numpy.pkl'
    if not os.path.exists(assets_path):
        return None
    return joblib.load(assets_path)

assets = load_numpy_model()

# ================== 3. 核心界面布局 ==================
col_left, spacer, col_right = st.columns([6.8, 0.2, 3.0])

# ----------------- 左侧：19个参数 6列布局区 -----------------
with col_left:
    st.markdown("<h4 style='color: #333;'>Seismic Failure Mode Probability Assessment</h4>", unsafe_allow_html=True)
    
    cols = st.columns([1.0, 2.5, 1.8, 1.2, 1.0, 1.5])
    headers = ["Parameter", "Description", "Distribution", "Mean", "COV", "Range"]
    for i, header_name in enumerate(headers):
        cols[i].markdown(f"<div style='text-align: center; color: #800020; font-size: 20px; font-weight: bold; margin-bottom: -15px;'>{header_name}</div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    params_config = [
        ("N", "N", "Number of pile rows along the loading direction", "2~4", 2.0, 4.0, 3.0, "Deterministic", 0.0, 1.0, "%.0f"),
        ("Dp", "D<sub>p</sub> (m)", "Pile diameter", "0.6~1.8", 0.6, 1.8, 1.2, "Normal", 0.10, 0.1, "%.2f"),
        ("rho_pl", "ρ<sub>pile,l</sub>", "Pile longitudinal reinforcement ratio", "0.005~0.015", 0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f"),
        ("alpha", "α", "Column axial load ratio", "0.05~0.25", 0.05, 0.25, 0.15, "Normal", 0.12, 0.01, "%.2f"),
        ("S_Dp", "S (D<sub>p</sub>)", "Pile spacing-to-diameter ratio", "2.5~3.5", 2.5, 3.5, 3.0, "Normal", 0.15, 0.1, "%.2f"),
        ("Dr", "D<sub>r</sub>", "Sand density", "0.35~0.75", 0.35, 0.75, 0.55, "Uniform", 0.21, 0.05, "%.2f"),
        ("SD", "SD (m)", "Scour depth", "0~6", 0.0, 6.0, 3.0, "Normal", 0.27, 0.5, "%.2f"),
        ("Hp_Dc", "H<sub>p</sub>/D<sub>c</sub>", "Column aspect ratio", "1~4", 1.0, 4.0, 2.5, "Normal", 0.26, 0.1, "%.2f"),
        ("Dc_Dp", "D<sub>c</sub> (D<sub>p</sub>)", "Pier-to-pile diameter ratio", "1.5~2.5", 1.5, 2.5, 2.0, "Normal", 0.10, 0.1, "%.2f"),
        ("rho_cl", "ρ<sub>column,l</sub>", "Pier longitudinal reinforcement ratio", "0.005~0.015", 0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f"),
        ("rho_ps", "ρ<sub>pile,s</sub>", "Pile transverse reinforcement ratio", "0.003~0.013", 0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f"),
        ("fyl", "f<sub>yl</sub> (MPa)", "Longitudinal rebar yield strength", "300~500", 300.0, 500.0, 400.0, "Lognormal", 0.106, 10.0, "%.0f"),
        ("fc", "f<sub>c</sub> (MPa)", "Concrete compressive strength", "20~50", 20.0, 50.0, 35.0, "Lognormal", 0.20, 1.0, "%.1f"),
        ("rho_cs", "ρ<sub>column,s</sub>", "Pier transverse reinforcement ratio", "0.003~0.013", 0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f"),
        ("Xt", "X<sub>t</sub> / X<sub>l</sub>", "Transverse-to-longitudinal rebar corrosion", "1~3", 1.0, 3.0, 2.0, "Normal", 0.29, 0.1, "%.2f"),
        ("Xl", "X<sub>l</sub>", "Corrosion level of longitudinal reinforcement", "0~0.30", 0.0, 0.30, 0.15, "Normal", 0.20, 0.01, "%.2f"),
        ("t", "t (m)", "Pier cover concrete thickness", "0.04~0.08", 0.04, 0.08, 0.06, "Normal", 0.20, 0.01, "%.2f"),
        ("d_l", "d<sub>l</sub> (m)", "Pier longitudinal reinforcement diameter", "0.018~0.032", 0.018, 0.032, 0.025, "Normal", 0.10, 0.001, "%.3f"),
        ("fyt", "f<sub>yt</sub> (MPa)", "Transverse rebar yield strength", "200~400", 200.0, 400.0, 300.0, "Lognormal", 0.106, 10.0, "%.0f")
    ]

    dist_options = ["Normal", "Lognormal", "Uniform", "Deterministic"]
    user_inputs = []
    
    for p_id, html_name, desc, rng, p_min, p_max, p_mean, p_dist, p_cov, p_step, p_format in params_config:
        c1, c2, c3, c4, c5, c6 = st.columns([1.0, 2.5, 1.8, 1.2, 1.0, 1.5])
        
        c1.markdown(f"<div style='text-align: center; color: #4a235a; font-weight: bold; font-family: Arial; padding-top: 8px;'>{html_name}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='text-align: center; color: #444444; font-size: 14px; padding-top: 8px; padding-left: 10px;'>{desc}</div>", unsafe_allow_html=True)
        
        with c3:
            dist_val = st.selectbox(label=p_id+"_dist", options=dist_options, index=dist_options.index(p_dist), label_visibility="collapsed")
        with c4:
            mean_val = st.number_input(label=p_id+"_mean", min_value=p_min, max_value=p_max, value=float(p_mean), step=p_step, format=p_format, label_visibility="collapsed")
        with c5:
            cov_disabled = (dist_val == "Deterministic")
            cov_val = st.number_input(label=p_id+"_cov", min_value=0.0, max_value=2.0, value=0.0 if cov_disabled else float(p_cov), step=0.05, format="%.2f", disabled=cov_disabled, label_visibility="collapsed")
            
        c6.markdown(f"<div style='text-align: center; color: #666666; font-size: 16px; padding-top: 8px;'>{rng}</div>", unsafe_allow_html=True)
        
        user_inputs.append({"mean": mean_val, "cov": cov_val, "dist": dist_val, "min": p_min, "max": p_max})

# ----------------- 右侧：控制与输出区 -----------------
with col_right:
    st.markdown("""
        <div style='text-align: right; color: #555555; line-height: 1.5; font-family: Arial; font-size: 14px; margin-top: 25px;'>
            Created by Jingcheng Wang, Associate Professor. Fuzhou University<br>Contact: Jingchengwang@fzu.edu.cn
        </div>
    """, unsafe_allow_html=True)
    st.write("")
    
    predict_clicked = st.button("Calculate (Monte Carlo Simulation)", type="primary", use_container_width=True)
    
    # 为下载按钮预留占位符
    download_placeholder = st.empty()
    st.write("")

    st.markdown("<h5 style='color: #333; font-family: Arial; font-weight: bold;'>Failure Mode Probabilities</h5>", unsafe_allow_html=True)

    def draw_progress_bar(label, percentage, color_hex):
        text_color = "white" if percentage > 50 else "#333333"
        html = f"""
        <div style="margin-bottom: 15px;">
            <div style="font-weight: bold; font-family: Arial; font-size: 14px; margin-bottom: 5px;">{label}</div>
            <div style="width: 100%; background-color: #f0f0f0; border-radius: 4px; border: 1px solid #ccc; position: relative; height: 28px;">
                <div style="width: {percentage}%; background-color: {color_hex}; height: 100%; border-radius: 3px; transition: width 0.6s ease-in-out;"></div>
                <div style="position: absolute; width: 100%; text-align: center; top: 0; left: 0; line-height: 28px; font-weight: bold; font-family: Arial; font-size: 13px; color: {text_color};">{percentage:.2f}%</div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    if 'mc_probs' not in st.session_state:
        st.session_state.mc_probs = [0.0, 0.0, 0.0]
    
    if predict_clicked:
        if assets is None:
            st.error("⚠️ 未检测到 model_assets_numpy.pkl。请将纯权重文件放在同一目录下。")
        else:
            weights = assets['weights']
            scaler = assets['scaler']
            label_names = assets['le'].classes_
            
            N_SAMPLES = 5000  
            samples = np.zeros((N_SAMPLES, 19))
            
            for i in range(19):
                mean, cov, dist = user_inputs[i]["mean"], user_inputs[i]["cov"], user_inputs[i]["dist"]
                p_min, p_max = user_inputs[i]["min"], user_inputs[i]["max"]
                std = mean * cov
                
                if dist == "Deterministic" or cov == 0:
                    s = np.full(N_SAMPLES, mean)
                elif dist == "Normal":
                    s = np.random.normal(mean, std, N_SAMPLES)
                elif dist == "Lognormal":
                    sigma2 = np.log(1 + (std/mean)**2)
                    mu = np.log(mean) - sigma2 / 2
                    s = np.random.lognormal(mu, np.sqrt(sigma2), N_SAMPLES)
                elif dist == "Uniform":
                    lower, upper = mean - np.sqrt(3) * std, mean + np.sqrt(3) * std
                    s = np.random.uniform(lower, upper, N_SAMPLES)
                
                samples[:, i] = np.clip(s, p_min, p_max)
            
            # =============== 纯 NumPy 极速矩阵前向传播 ===============
            a = scaler.transform(samples)
            for w, b in weights[:-1]:
                z = np.dot(a, w) + b
                a = np.maximum(0, z) # ReLU
            w_out, b_out = weights[-1]
            z_out = np.dot(a, w_out) + b_out
            
            # 获取每行的最大概率索引作为分类结果
            predictions = np.argmax(z_out, axis=1)
            
            # 统计概率
            probs_dict = {}
            for idx, name in enumerate(label_names):
                count = np.sum(predictions == idx)
                probs_dict[name] = count / N_SAMPLES
            
            st.session_state.mc_probs = [probs_dict.get('PFF', 0.0), probs_dict.get('FFF', 0.0), probs_dict.get('PSF', 0.0)]

            # =============== 后台生成 Excel 文件数据 ===============
            excel_columns = ["N", "Dp (m)", "rho_pile,l", "alpha", "S (Dp)", "Dr", "SD (m)", "Hp/Dc", 
                             "Dc (Dp)", "rho_column,l", "rho_pile,s", "fyl (MPa)", "fc (MPa)", 
                             "rho_column,s", "Xt/Xl", "Xl", "t (m)", "dl (m)", "fyt (MPa)"]
            
            df_samples = pd.DataFrame(samples, columns=excel_columns)
            df_samples['Failure_Mode'] = [label_names[i] for i in predictions]
            
            counts = [int(probs_dict.get(lab, 0) * N_SAMPLES) for lab in label_names]
            df_stats = pd.DataFrame({
                'Failure_Mode': label_names,
                'Count': counts,
                'Probability': [f"{probs_dict.get(lab, 0.0)*100:.2f}%" for lab in label_names]
            })
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_samples.to_excel(writer, sheet_name='Samples', index=False)
                df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            st.session_state.excel_data = output.getvalue()

    if 'excel_data' in st.session_state:
        with download_placeholder.container():
            st.download_button(
                label="Save results",
                data=st.session_state.excel_data,
                file_name="Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )

    probs = st.session_state.mc_probs
    draw_progress_bar("PFF (Pier Flexure Failure)", probs[0] * 100, "#0078d4") 
    draw_progress_bar("FFF (Foundation Flexure Failure)", probs[1] * 100, "#008000") 
    draw_progress_bar("PSF (Pier Shear Failure)", probs[2] * 100, "#d83b01") 

    st.write("---")
    st.markdown("<h5 style='color: #333; font-family: Arial; font-weight: bold; margin-bottom: 10px;'>Parameter Schematic</h5>", unsafe_allow_html=True)
    try:
        st.image("structure.png", use_container_width=True)
    except:
        st.markdown("""
        <div style='border: 1px dashed #ccc; padding: 40px; text-align: center; color: #999; font-family: Arial;'>
            Structure Image<br>(structure.png not found)
        </div>
        """, unsafe_allow_html=True)
