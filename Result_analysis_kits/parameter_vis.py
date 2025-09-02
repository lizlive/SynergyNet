import json
import plotly.express as px
import os


def parallel_coordinates(metric_show, cfg, df, save_path):
    with open(cfg.tune_hp_json) as f:
        raw_hp_json = json.load(f)

    columns_to_plot = ["config/" + key for key in raw_hp_json.keys()]
    columns_to_plot.append(metric_show)
    data_to_plot = df[columns_to_plot]

    # 绘制带渐变色的平行坐标图
    fig = px.parallel_coordinates(
        data_to_plot,
        color=metric_show,
        color_continuous_scale=px.colors.sequential.YlOrRd,
        range_color=[df[metric_show].min(), df[metric_show].max()]  # 手动设置颜色条范围
        # color_continuous_midpoint=df[metric_show].median()
    )
    fig.write_image(os.path.join(save_path, f"{metric_show}_parallel_coordinates_plot.png"))
    fig.write_html(os.path.join(save_path,  f"{metric_show}_parallel_coordinates_plot.html"))

