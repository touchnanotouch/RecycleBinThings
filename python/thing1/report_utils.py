import os

from jinja2 import Environment, FileSystemLoader


def collect_report_data(
    x, gamma_vals, eigenfunctions, residuals, alpha, beta, p, n, img_path, html_dir=None
):
    # Дисперсии и центры масс
    dispersions = []
    centers = []
    dx = x[1] - x[0]
    for u in eigenfunctions:
        norm = (u**2).sum() * dx
        if norm > 0:
            prob_density = u**2 / norm
            x_mean = (x * prob_density).sum() * dx
            var = ((x - x_mean)**2 * prob_density).sum() * dx
            dispersions.append(var)
            centers.append(x_mean)
        else:
            dispersions.append(float("nan"))
            centers.append(float("nan"))

    # Вычисляем относительный путь к картинке относительно папки с report.html
    if html_dir is not None:
        rel_img_path = os.path.relpath(img_path, html_dir)
    else:
        rel_img_path = os.path.basename(img_path)

    return {
        "beta": beta,
        "alpha": alpha,
        "gamma_vals": gamma_vals,
        "residuals": residuals,
        "dispersions": dispersions,
        "centers": centers,
        "img_path": rel_img_path
    }


def generate_html_report(runs, n, p, h, num_eigen, template_path, output_path):
    env = Environment(loader=FileSystemLoader(searchpath="."))
    template = env.get_template(template_path)
    html = template.render(
        runs=runs,
        n=n,
        p=p,
        h=h,
        num_eigen=num_eigen
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)