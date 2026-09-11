import os
import time
import streamlit as st

from gui.image_utils import load_image_safely
from gui.monitor_utils import collect_checkpoint_status, format_seconds, load_training_metrics, parse_training_progress, read_monitor_log, resolve_result_dir
from gui.training_service import start_tensorboard

MAX_MONITOR_LOG_LINES = 2000


def downsample_metrics_for_chart(data: list[dict], thresholds: list[tuple[int, int]]) -> tuple[list[dict], int]:
    """按训练轮次自适应抽样，降低前端图表渲染压力。"""
    if not data:
        return [], 1

    max_episode = int(data[-1].get("episode", len(data)) or len(data))
    stride = 1
    for episode_limit, candidate_stride in thresholds:
        if max_episode <= episode_limit:
            stride = candidate_stride
            break
    else:
        stride = thresholds[-1][1] if thresholds else 1

    if stride == 1:
        return data, stride

    sampled_rows = [data[0]]
    last_bucket = None
    for row in data[1:]:
        episode = int(row.get("episode", 0) or 0)
        bucket = max(0, (episode - 1) // stride)
        if sampled_rows and bucket == last_bucket:
            sampled_rows[-1] = row
        else:
            sampled_rows.append(row)
            last_bucket = bucket

    if sampled_rows and sampled_rows[-1] != data[-1]:
        sampled_rows.append(data[-1])
    return sampled_rows, stride


def render_metric_line_chart(data: list[dict], title: str, y_title: str, series: dict[str, str], height: int = 220) -> None:
    """使用轻量配置渲染速度曲线，避免额外数据处理依赖。"""
    if not data:
        return
    chart_data = []
    for row in data:
        episode = row.get("episode")
        if episode is None:
            continue
        for field_name, display_name in series.items():
            value = row.get(field_name)
            if value is None:
                continue
            chart_data.append({
                "episode": episode,
                "series": display_name,
                "数值": value,
            })
    if not chart_data:
        return

    st.vega_lite_chart(
        {
            "data": {"values": chart_data},
            "title": title,
            "mark": {"type": "line", "point": False},
            "encoding": {
                "x": {"field": "episode", "type": "quantitative", "title": "训练轮次"},
                "y": {"field": "数值", "type": "quantitative", "title": y_title},
                "color": {"field": "series", "type": "nominal", "title": ""},
                "tooltip": [
                    {"field": "episode", "type": "quantitative", "title": "轮次"},
                    {"field": "series", "type": "nominal", "title": "指标"},
                    {"field": "数值", "type": "quantitative", "title": y_title, "format": ".4f"},
                ],
            },
            "height": height,
        },
        use_container_width=True,
    )


def render_monitor_page(log_content: str, agent_name: str, configured_episodes: int, render_enabled: bool) -> None:
    """渲染训练监控页。"""
    st.subheader("训练监控")
    st.caption("日志文件直接来自训练进程输出，并结合结果目录展示训练进度。")

    log_content = read_monitor_log(log_content)
    all_log_lines = log_content.splitlines()
    log_action_col1, log_action_col2 = st.columns([1.2, 4.0])
    with log_action_col1:
        st.download_button(
            label="导出当前日志",
            data=log_content,
            file_name=f"training_log_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with log_action_col2:
        st.caption(f"日志窗口仅显示最近 {MAX_MONITOR_LOG_LINES} 行。")

    progress = parse_training_progress(log_content)
    if progress.total_episodes <= 0 and configured_episodes > 0:
        progress.total_episodes = configured_episodes

    result_dir = resolve_result_dir(log_content, agent_name)
    checkpoint_status = collect_checkpoint_status(result_dir, use_live_render=render_enabled)
    metrics_df = load_training_metrics(result_dir)
    speed_chart_metrics_df, speed_chart_stride = downsample_metrics_for_chart(
        metrics_df,
        thresholds=[(200, 1), (1000, 100), (5000, 100), (10**9, 200)],
    )
    reward_chart_metrics_df, reward_chart_stride = downsample_metrics_for_chart(
        metrics_df,
        thresholds=[(200, 1), (1000, 20), (5000, 100), (10**9, 200)],
    )
    if metrics_df:
        latest_metrics = metrics_df[-1]
        progress.current_episode = max(progress.current_episode, int(latest_metrics.get("episode", 0) or 0))
        progress.total_episodes = max(progress.total_episodes, int(latest_metrics.get("total_episodes", 0) or 0))
    display_total_episodes = progress.total_episodes or configured_episodes or 0
    progress_percent = 0.0
    progress_value = 0
    if display_total_episodes > 0:
        progress_percent = min(100.0, max(0.0, (progress.current_episode / display_total_episodes) * 100.0))
        progress_value = min(100, max(0, int(round(progress_percent))))
        if progress.current_episode > 0 and progress_value == 0:
            progress_value = 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("当前轮次", f"{progress.current_episode}/{display_total_episodes}")
    with col2:
        st.metric("总步数", progress.total_step if progress.total_step is not None else "-")
    with col3:
        st.metric("平均奖励", f"{progress.mean_reward:.3f}" if progress.mean_reward is not None else "-")
    with col4:
        st.metric("检查点数量", checkpoint_status["checkpoint_count"])

    time_col1, time_col2, time_col3, time_col4 = st.columns(4)
    with time_col1:
        st.metric("本轮耗时", format_seconds(progress.episode_time_seconds))
    with time_col2:
        st.metric("本轮步均耗时", format_seconds(progress.step_time_seconds, precision=4))
    with time_col3:
        st.metric("累计轮均耗时", format_seconds(progress.avg_episode_time_seconds))
    with time_col4:
        st.metric("累计步均耗时", format_seconds(progress.avg_step_time_seconds, precision=4))

    st.progress(progress_value, text=f"训练进度 {progress.current_episode}/{display_total_episodes} ({progress_percent:.2f}%)")

    st.markdown("### TensorBoard")
    tensorboard_logdir = os.path.join(checkpoint_status["result_dir"], "tensorboard") if checkpoint_status["result_dir"] else ""
    tb_col1, tb_col2, tb_col3 = st.columns([1.0, 1.0, 3.0])
    with tb_col1:
        if st.button("启动 TensorBoard", use_container_width=True, disabled=not os.path.isdir(tensorboard_logdir)):
            success, message = start_tensorboard(tensorboard_logdir)
            if success:
                st.success(f"TensorBoard 已启动: {message}")
            else:
                st.error(message)
    with tb_col2:
        if st.session_state.get("tensorboard_url"):
            st.link_button("打开 TensorBoard", st.session_state.tensorboard_url, use_container_width=True)
    with tb_col3:
        st.caption(
            f"日志目录: {tensorboard_logdir if tensorboard_logdir and os.path.isdir(tensorboard_logdir) else '尚未生成'}"
        )

    st.markdown("### 速度趋势")
    if metrics_df:
        trend_col1, trend_col2, trend_col3 = st.columns(3)
        with trend_col1:
            st.metric("当前训练速度", f"{latest_metrics['steps_per_second']:.2f} step/s")
        with trend_col2:
            st.metric("累计训练速度", f"{latest_metrics['avg_steps_per_second']:.2f} step/s")
        with trend_col3:
            st.metric("图表采样点", f"{len(speed_chart_metrics_df)}/{len(metrics_df)}")

        if speed_chart_stride > 1:
            st.caption(f"速度越高越好。为降低网页卡顿，图表已按训练轮次抽样显示：当前每 {speed_chart_stride} 轮保留一个点。")
        else:
            st.caption("速度越高越好，用于观察训练是否因为渲染、保存、奖励计算或资源竞争而变慢。")
        render_metric_line_chart(
            speed_chart_metrics_df,
            title="训练速度变化",
            y_title="step/s",
            series={
                "steps_per_second": "当前速度",
                "avg_steps_per_second": "累计平均速度",
            },
        )

        if speed_chart_stride > 1:
            st.caption(f"每步耗时越低越好。当前图表同样按每 {speed_chart_stride} 轮抽样，完整指标仍保留在 CSV 和明细中。")
        else:
            st.caption("每步耗时越低越好，适合观察优化是否生效。")
        render_metric_line_chart(
            speed_chart_metrics_df,
            title="每步耗时变化",
            y_title="ms",
            series={
                "step_milliseconds": "当前每步耗时",
                "avg_step_milliseconds": "累计平均每步耗时",
            },
        )

        st.markdown("### 奖励趋势")
        reward_col1, reward_col2, reward_col3 = st.columns(3)
        with reward_col1:
            st.metric("当前平均奖励", f"{latest_metrics['mean_reward']:.3f}")
        with reward_col2:
            st.metric("当前中位奖励", f"{latest_metrics['median_reward']:.3f}")
        with reward_col3:
            st.metric("奖励图采样点", f"{len(reward_chart_metrics_df)}/{len(metrics_df)}")

        if reward_chart_stride > 1:
            st.caption(f"奖励值用于观察训练目标是否持续改善。当前奖励图按每 {reward_chart_stride} 轮保留一个点，抽样比速度图更细。")
        else:
            st.caption("奖励值用于观察训练目标是否持续改善。")
        render_metric_line_chart(
            reward_chart_metrics_df,
            title="奖励值变化",
            y_title="reward",
            series={
                "mean_reward": "平均奖励",
                "median_reward": "中位奖励",
            },
        )

        with st.expander("查看速度指标明细", expanded=False):
            detail_lines = []
            for row in metrics_df[-20:]:
                detail_lines.append(
                    "Episode {episode}: speed={steps_per_second:.2f} step/s, avg_speed={avg_steps_per_second:.2f} step/s, ep={episode_seconds:.2f}s, step={step_milliseconds:.2f}ms, reward={mean_reward:.3f}".format(
                        **row
                    )
                )
            st.code("\n".join(detail_lines) if detail_lines else "暂无明细数据", language="text")
    else:
        st.info("暂未检测到训练速度指标 CSV。训练完成至少一轮后，这里会自动显示速度变化图。")

    st.markdown("### 结果状态")
    st.caption(f"结果目录: {checkpoint_status['result_dir'] or '尚未生成'}")
    metrics_csv_path = os.path.join(checkpoint_status["result_dir"], "training_metrics.csv") if checkpoint_status["result_dir"] else ""
    st.caption(f"速度指标文件: {metrics_csv_path if metrics_csv_path and os.path.isfile(metrics_csv_path) else '未找到'}")

    state_col1, state_col2 = st.columns(2)
    with state_col1:
        st.caption(f"best_model: {'已保存' if checkpoint_status['best_model_exists'] else '未保存'}")
        st.caption(f"final_model: {'已保存' if checkpoint_status['final_model_exists'] else '未保存'}")
    with state_col2:
        st.caption(f"最新 checkpoint: {checkpoint_status['latest_checkpoint'] or '暂无'}")
        st.caption(f"最新布局图: {checkpoint_status['latest_image'] or '暂无'}")

    st.markdown("### 实时渲染")
    preview_image = checkpoint_status["preview_image"]
    if preview_image and os.path.isfile(preview_image):
        if render_enabled:
            st.caption(
                "训练运行中优先显示按步长与时间间隔更新的 live_render_时间戳.png；若未生成，则回退到按配置间隔更新的 latest_layout.png 或最近保存的检查点图像。"
            )
        else:
            st.caption("当前未开启渲染，直接显示最新保存的 latest_layout.png 或最近一次奖励达标保存的布局图。")
        image = load_image_safely(preview_image)
        if image is not None:
            col_image, col_option = st.columns([4, 1], vertical_alignment="top")
            with col_option:
                fit_width = st.checkbox("适配页面宽度", value=False, key="monitor_fit_width")
                image_width = st.slider("显示宽度", min_value=360, max_value=1100, value=700, step=20, disabled=fit_width, key="monitor_image_width")
            with col_image:
                if fit_width:
                    st.image(image, caption=preview_image, use_container_width=True)
                else:
                    st.image(image, caption=preview_image, width=image_width)
        else:
            st.warning("实时渲染图正在更新，当前帧暂不可读，稍后页面会自动刷新。")
    else:
        if render_enabled:
            st.info("暂未检测到实时渲染图像。开启渲染并启动训练后，这里会自动刷新显示。")
        else:
            st.info("暂未检测到已保存的最新布局图。训练完成至少一轮保存后，这里会自动显示。")

    recent_log_lines = all_log_lines[-MAX_MONITOR_LOG_LINES:]
    reversed_recent_log = "\n".join(recent_log_lines[::-1])
    st.markdown("实时日志（最新在上）")
    st.code(reversed_recent_log or "暂无日志", language="text", height=640, wrap_lines=True)
