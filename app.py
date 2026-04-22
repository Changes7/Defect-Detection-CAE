import streamlit as st
import datetime
import time
import sqlite3
import hashlib

# ==========================================
# 0. 全局页面配置 (必须是第一行)
# ==========================================
st.set_page_config(page_title="工业质检云平台", page_icon="🛡️", layout="wide")

# ==========================================
# 1. 数据库与密码安全模块
# ==========================================
# 密码加密函数（展示企业级安全意识）
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 初始化用户数据库
def init_auth_db():
    conn = sqlite3.connect('auth.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT NOT NULL
        )
    ''')
    
    # 预置一个超级管理员账号 (如果数据库是空的)
    c.execute('SELECT * FROM users WHERE username="admin"')
    if not c.fetchone():
        c.execute('INSERT INTO users (username, password, role, name) VALUES (?, ?, ?, ?)',
                  ('admin', hash_password('123'), '管理员', '超级管理员'))
    conn.commit()
    conn.close()

# 验证登录
def verify_user(username, password):
    conn = sqlite3.connect('auth.db')
    c = conn.cursor()
    c.execute('SELECT password, role, name FROM users WHERE username=?', (username,))
    record = c.fetchone()
    conn.close()
    
    if record and record[0] == hash_password(password):
        return {"role": record[1], "name": record[2]}
    return None

# 注册新用户
def create_user(username, password, role, name):
    try:
        conn = sqlite3.connect('auth.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password, role, name) VALUES (?, ?, ?, ?)',
                  (username, hash_password(password), role, name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False # 用户名已存在

# 初始化系统
init_auth_db()

# 初始化 Session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# ==========================================
# 2. 拦截网关：未登录状态展示认证中心
# ==========================================
if not st.session_state.logged_in:
    # 隐藏左侧菜单
    st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>🏭 CAE-BN 工业质检平台</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>企业级统一身份认证网关 (SSO)</p>", unsafe_allow_html=True)
        
        # 使用选项卡分离登录和注册
        tab_login, tab_register = st.tabs(["🔐 员工登录", "📝 新员工注册"])
        
        # --- 登录面板 ---
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("工号 / 用户名")
                password = st.text_input("密码", type="password")
                submit = st.form_submit_button("安全登录", type="primary", use_container_width=True)
                
                if submit:
                    if not username or not password:
                        st.error("⚠️ 请填写完整信息！")
                    else:
                        user_info = verify_user(username, password)
                        if user_info:
                            st.session_state.logged_in = True
                            st.session_state.role = user_info["role"]
                            st.session_state.name = user_info["name"]
                            st.success(f"✅ 身份核验通过！欢迎，{st.session_state.name}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ 账号或密码错误！")
                            
        # --- 注册面板 ---
        with tab_register:
            with st.form("register_form"):
                reg_user = st.text_input("设置登录工号 (英文字母或数字)")
                reg_name = st.text_input("真实姓名")
                reg_pwd = st.text_input("设置密码", type="password")
                reg_role = st.selectbox("申请权限组", ["操作员", "管理员"])
                # 管理员注册需要特殊验证码
                reg_secret = st.text_input("管理员邀请码 (注册操作员无需填写，管理员填: AGV2026)", type="password")
                
                reg_submit = st.form_submit_button("注册账号", use_container_width=True)
                
                if reg_submit:
                    if not reg_user or not reg_pwd or not reg_name:
                        st.error("⚠️ 请完整填写带 * 的必填项！")
                    elif reg_role == "管理员" and reg_secret != "AGV2026":
                        st.error("⛔ 邀请码错误！您无权注册管理员账号。")
                    else:
                        success = create_user(reg_user, reg_pwd, reg_role, reg_name)
                        if success:
                            st.success(f"🎉 注册成功！请切换到【登录】面板进行登录。")
                        else:
                            st.error("❌ 该工号已被注册，请更换！")

# ==========================================
# 3. 授权状态：展示系统大盘界面
# ==========================================
else:
    # 侧边栏显示用户信息
    st.sidebar.info(f"👤 当前在线: {st.session_state.name}\n\n🛡️ 权限组: **{st.session_state.role}**")
    if st.sidebar.button("🚪 安全退出系统", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # ---- 系统主页内容 ----
    st.markdown("""
    <style>
        .main-title { font-size: 2.5rem; font-weight: 800; color: #0F172A; margin-bottom: 0px; }
        .sub-title { font-size: 1.1rem; color: #64748B; margin-top: 5px; margin-bottom: 30px; }
        .card { background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E2E8F0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">🏭 CAE-BN 工业表面智能质检平台</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">基于深度无监督学习的高精度机器视觉检测系统</p>', unsafe_allow_html=True)

    st.markdown("### 🖥️ 核心引擎状态")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label="视觉推理核心", value="就绪", delta="PyTorch CUDA")
    with col2: st.metric(label="认证数据库", value="安全加密", delta="SHA-256")
    with col3: st.metric(label="当前权限等级", value=st.session_state.role)
    with col4: st.metric(label="系统时间", value=datetime.datetime.now().strftime("%H:%M"), delta_color="off")
    st.markdown("---")
    
    st.info("👈 请点击左侧边栏进入具体业务模块。")