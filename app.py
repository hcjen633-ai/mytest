import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# 設定頁面
st.set_page_config(
    page_title="Association Rule Mining 分析工具",
    page_icon="🏠",
    layout="wide"
)

# 標題
st.title("🏠 房屋資料 Association Rule Mining 分析工具")
st.markdown("---")

# 側邊欄說明
with st.sidebar:
    st.header("📖 使用說明")
    st.markdown("""
    ### 步驟：
    1. 上傳您的 CSV 檔案
    2. 選擇要分析的欄位
    3. 調整參數
    4. 查看分析結果
    
    ### 關於 Association Rules:
    - **Support**: 項目集出現的頻率
    - **Confidence**: 規則的可信度
    - **Lift**: 規則的提升度（>1 表示正相關）
    
    ### 參數建議：
    - Support: 0.1 - 0.3
    - Confidence: 0.5 - 0.8
    """)
    
    st.markdown("---")
    st.markdown("💡 **提示**: 數值越小，找到的規則越多")

# 主要內容
tab1, tab2, tab3 = st.tabs(["📁 資料上傳與設定", "📊 分析結果", "📈 視覺化"])

with tab1:
    # 檔案上傳
    st.header("📁 上傳資料")
    uploaded_file = st.file_uploader(
        "選擇 CSV 檔案",
        type=['csv'],
        help="請上傳包含數值欄位的 CSV 檔案"
    )
    
    # 提供範例檔案
    st.markdown("**或使用內建範例資料**")
    use_sample = st.checkbox("使用 USA_Housing_300.csv 範例資料")
    
    if use_sample:
        # 讀取上傳的範例檔案
        df = pd.read_csv('/mnt/user-data/uploads/USA_Housing_300.csv')
        st.success("✅ 已載入範例資料！")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ 檔案上傳成功！")
    else:
        df = None
        st.info("👆 請上傳 CSV 檔案或使用範例資料")
    
    if df is not None:
        # 顯示原始資料
        st.subheader("📋 原始資料預覽")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.write(f"**資料維度**: {df.shape[0]} 列 × {df.shape[1]} 欄")
        
        # 欄位選擇
        st.subheader("⚙️ 選擇要分析的欄位")
        
        # 只顯示數值型欄位
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) == 0:
            st.error("❌ 找不到數值型欄位！請確認您的資料格式。")
        else:
            selected_columns = st.multiselect(
                "選擇欄位（可多選）",
                numeric_columns,
                default=numeric_columns[:4] if len(numeric_columns) >= 4 else numeric_columns,
                help="選擇要進行關聯分析的數值欄位"
            )
            
            if len(selected_columns) > 0:
                # 離散化參數
                st.subheader("🔧 離散化設定")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    n_bins = st.selectbox(
                        "分組數量",
                        options=[2, 3, 4, 5],
                        index=1,
                        help="將連續數值分成幾組"
                    )
                
                with col2:
                    binning_method = st.selectbox(
                        "分組方法",
                        options=["等頻分組 (qcut)", "等距分組 (cut)"],
                        index=0,
                        help="等頻：每組數量相近 / 等距：每組範圍相等"
                    )
                
                # 自訂標籤選項
                st.markdown("**類別標籤**")
                label_options = {
                    2: ["低", "高"],
                    3: ["低", "中", "高"],
                    4: ["很低", "低", "高", "很高"],
                    5: ["很低", "低", "中", "高", "很高"]
                }
                
                # 執行離散化
                if st.button("🚀 開始離散化", type="primary"):
                    with st.spinner("處理中..."):
                        try:
                            transactions = pd.DataFrame()
                            
                            for col in selected_columns:
                                # 根據選擇的方法進行分組
                                if binning_method == "等頻分組 (qcut)":
                                    transactions[col] = pd.qcut(
                                        df[col], 
                                        q=n_bins, 
                                        labels=[f"{col}_{label}" for label in label_options[n_bins]],
                                        duplicates='drop'
                                    )
                                else:
                                    transactions[col] = pd.cut(
                                        df[col], 
                                        bins=n_bins, 
                                        labels=[f"{col}_{label}" for label in label_options[n_bins]]
                                    )
                            
                            # 儲存到 session state
                            st.session_state['transactions'] = transactions
                            st.session_state['selected_columns'] = selected_columns
                            
                            st.success("✅ 離散化完成！請到「分析結果」頁籤查看。")
                            
                            # 顯示離散化結果預覽
                            st.subheader("🔍 離散化結果預覽")
                            st.dataframe(transactions.head(10), use_container_width=True)
                            
                            # 顯示每個類別的分佈
                            st.subheader("📊 類別分佈")
                            
                            cols_per_row = 2
                            cols = st.columns(cols_per_row)
                            
                            for idx, col in enumerate(selected_columns):
                                with cols[idx % cols_per_row]:
                                    value_counts = transactions[col].value_counts()
                                    st.write(f"**{col}**")
                                    st.bar_chart(value_counts)
                        
                        except Exception as e:
                            st.error(f"❌ 離散化過程發生錯誤: {str(e)}")

with tab2:
    st.header("📊 Association Rule Mining 結果")
    
    if 'transactions' not in st.session_state:
        st.info("👈 請先在「資料上傳與設定」頁籤完成資料離散化")
    else:
        transactions = st.session_state['transactions']
        
        # 參數設定
        st.subheader("⚙️ 挖掘參數")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_support = st.slider(
                "Minimum Support",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="項目集至少要出現的比例"
            )
        
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.3,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="規則的最低可信度"
            )
        
        with col3:
            min_lift = st.slider(
                "Minimum Lift",
                min_value=1.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="規則的最低提升度"
            )
        
        # 執行分析
        if st.button("🔍 開始挖掘關聯規則", type="primary"):
            with st.spinner("正在分析中..."):
                try:
                    # 準備交易資料格式
                    records = []
                    for _, row in transactions.iterrows():
                        records.append([str(val) for val in row.values if pd.notna(val)])
                    
                    # 轉換成 one-hot encoding
                    te = TransactionEncoder()
                    te_ary = te.fit(records).transform(records)
                    df_basket = pd.DataFrame(te_ary, columns=te.columns_)
                    
                    # 執行 Apriori
                    frequent_itemsets = apriori(df_basket, min_support=min_support, use_colnames=True)
                    
                    if len(frequent_itemsets) == 0:
                        st.warning("⚠️ 沒有找到符合條件的頻繁項目集，請降低 Support 值")
                    else:
                        st.success(f"✅ 找到 {len(frequent_itemsets)} 個頻繁項目集")
                        
                        # 產生關聯規則
                        if len(frequent_itemsets) > 0:
                            rules = association_rules(
                                frequent_itemsets, 
                                metric="confidence", 
                                min_threshold=min_confidence
                            )
                            
                            # 篩選 lift
                            rules = rules[rules['lift'] >= min_lift]
                            
                            if len(rules) == 0:
                                st.warning("⚠️ 沒有找到符合條件的關聯規則，請調整參數")
                            else:
                                # 儲存結果
                                st.session_state['rules'] = rules
                                st.session_state['frequent_itemsets'] = frequent_itemsets
                                
                                st.success(f"✅ 找到 {len(rules)} 條關聯規則！")
                                
                                # 排序規則
                                rules_sorted = rules.sort_values('lift', ascending=False)
                                
                                # 格式化顯示
                                display_rules = rules_sorted.copy()
                                display_rules['antecedents'] = display_rules['antecedents'].apply(
                                    lambda x: ', '.join(list(x))
                                )
                                display_rules['consequents'] = display_rules['consequents'].apply(
                                    lambda x: ', '.join(list(x))
                                )
                                
                                # 選擇要顯示的欄位
                                display_df = display_rules[[
                                    'antecedents', 'consequents', 
                                    'support', 'confidence', 'lift'
                                ]].copy()
                                
                                display_df.columns = ['前項 (IF)', '後項 (THEN)', 'Support', 'Confidence', 'Lift']
                                display_df = display_df.round(3)
                                
                                # 顯示規則表格
                                st.subheader(f"📋 關聯規則列表 (共 {len(display_df)} 條)")
                                st.dataframe(
                                    display_df.reset_index(drop=True),
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # 下載按鈕
                                csv = display_df.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    label="📥 下載規則 CSV",
                                    data=csv,
                                    file_name="association_rules.csv",
                                    mime="text/csv"
                                )
                                
                                # 規則統計
                                st.subheader("📈 規則統計")
                                
                                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                
                                with stat_col1:
                                    st.metric("規則總數", len(rules))
                                
                                with stat_col2:
                                    st.metric("平均 Support", f"{rules['support'].mean():.3f}")
                                
                                with stat_col3:
                                    st.metric("平均 Confidence", f"{rules['confidence'].mean():.3f}")
                                
                                with stat_col4:
                                    st.metric("平均 Lift", f"{rules['lift'].mean():.3f}")
                                
                                # Top 規則
                                st.subheader("🏆 Top 10 規則（依 Lift 排序）")
                                
                                for idx, row in display_df.head(10).iterrows():
                                    with st.expander(f"規則 {idx + 1}: {row['前項 (IF)']} → {row['後項 (THEN)']}"):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Support", row['Support'])
                                        with col2:
                                            st.metric("Confidence", row['Confidence'])
                                        with col3:
                                            st.metric("Lift", row['Lift'])
                                        
                                        st.markdown(f"""
                                        **解讀**:
                                        - 在所有交易中，有 {row['Support']*100:.1f}% 同時包含這兩個條件
                                        - 當出現「{row['前項 (IF)']}」時，有 {row['Confidence']*100:.1f}% 的機會出現「{row['後項 (THEN)']}」
                                        - 相比隨機情況，這個規則的關聯性提升了 {row['Lift']:.2f} 倍
                                        """)
                
                except Exception as e:
                    st.error(f"❌ 分析過程發生錯誤: {str(e)}")
                    st.exception(e)

with tab3:
    st.header("📈 視覺化分析")
    
    if 'rules' not in st.session_state:
        st.info("👈 請先在「分析結果」頁籤完成關聯規則挖掘")
    else:
        rules = st.session_state['rules']
        
        # 視覺化選項
        viz_option = st.selectbox(
            "選擇視覺化類型",
            [
                "Support vs Confidence (氣泡圖)",
                "規則散佈圖矩陣",
                "Lift 分佈直方圖",
                "Support vs Lift",
                "Top 規則條形圖"
            ]
        )
        
        if viz_option == "Support vs Confidence (氣泡圖)":
            st.subheader("Support vs Confidence (氣泡大小 = Lift)")
            
            fig = px.scatter(
                rules,
                x='support',
                y='confidence',
                size='lift',
                color='lift',
                hover_data=['antecedents', 'consequents'],
                color_continuous_scale='Viridis',
                title='Association Rules - Support vs Confidence'
            )
            
            fig.update_layout(
                xaxis_title='Support',
                yaxis_title='Confidence',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "規則散佈圖矩陣":
            st.subheader("規則指標關係矩陣")
            
            fig = px.scatter_matrix(
                rules,
                dimensions=['support', 'confidence', 'lift'],
                color='lift',
                color_continuous_scale='Viridis',
                title='規則指標散佈圖矩陣'
            )
            
            fig.update_layout(height=700)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Lift 分佈直方圖":
            st.subheader("Lift 值分佈")
            
            fig = px.histogram(
                rules,
                x='lift',
                nbins=30,
                title='Lift 分佈直方圖',
                labels={'lift': 'Lift', 'count': '規則數量'}
            )
            
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Support vs Lift":
            st.subheader("Support vs Lift")
            
            fig = px.scatter(
                rules,
                x='support',
                y='lift',
                size='confidence',
                color='confidence',
                hover_data=['antecedents', 'consequents'],
                color_continuous_scale='Blues',
                title='Support vs Lift (氣泡大小 = Confidence)'
            )
            
            fig.update_layout(
                xaxis_title='Support',
                yaxis_title='Lift',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Top 規則條形圖":
            st.subheader("Top 20 規則（依 Lift 排序）")
            
            top_rules = rules.nlargest(20, 'lift').copy()
            top_rules['rule'] = top_rules.apply(
                lambda x: f"{', '.join(list(x['antecedents']))} → {', '.join(list(x['consequents']))}",
                axis=1
            )
            
            fig = px.bar(
                top_rules.sort_values('lift'),
                y='rule',
                x='lift',
                orientation='h',
                color='lift',
                color_continuous_scale='Reds',
                title='Top 20 規則 (依 Lift)',
                labels={'lift': 'Lift', 'rule': '規則'}
            )
            
            fig.update_layout(height=800)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 額外的統計圖表
        st.markdown("---")
        st.subheader("📊 額外統計")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Support 箱型圖
            fig_box_support = px.box(
                rules,
                y='support',
                title='Support 分佈箱型圖',
                labels={'support': 'Support'}
            )
            st.plotly_chart(fig_box_support, use_container_width=True)
        
        with col2:
            # Confidence 箱型圖
            fig_box_conf = px.box(
                rules,
                y='confidence',
                title='Confidence 分佈箱型圖',
                labels={'confidence': 'Confidence'}
            )
            st.plotly_chart(fig_box_conf, use_container_width=True)

# 頁尾
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 Association Rule Mining 教學工具</p>
    <p>Powered by Streamlit | Made with ❤️</p>
</div>
""", unsafe_allow_html=True)
