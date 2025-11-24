#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
  递归自适应辛普森积分实现

  保持函数签名:
    double simpsons_rule(double (*f)(double), double a, double b, int n, double eps)

  说明:
  - n 只作为“最小分段数下限”。如果 n>1，则把区间均分为 n 段，对每一段分别做自适应积分，并把总误差 eps 平均分配到各段（每段 eps/n）。
  - 当 n <= 1 时，对整个区间直接进行自适应积分。
  - eps 默认为 1e-8（程序中提供了一个便捷包装函数 simpsons_rule_default）。
  - 算法基于标准的自适应辛普森法（使用 Richardson 外推修正 diff/15）。
  - 为避免无限递归，设置了最大递归深度（MAX_RECURSION_DEPTH）。
*/

#define MAX_RECURSION_DEPTH 50

/* 计算区间 [a,b] 上的单段辛普森近似，传入端点函数值以避免重复计算 */
static double simpson_segment(double fa, double fm, double fb, double a, double b)
{
    return (b - a) * (fa + 4.0 * fm + fb) / 6.0;
}

/* 递归自适应辅助函数
   参数:
     f    - 被积函数
     a,b  - 当前子区间
     eps  - 当前允许误差
     S    - 在 [a,b] 上已计算的辛普森估计
     fa,f b,fm - 已知的函数值 f(a), f(b), f(m)
     depth - 剩余允许递归深度
*/
static double adaptive_simpson_recursive(double (*f)(double),
                                         double a, double b,
                                         double eps,
                                         double S,
                                         double fa, double fb, double fm,
                                         int depth)
{
    double m = 0.5 * (a + b);
    double lm = 0.5 * (a + m);
    double rm = 0.5 * (m + b);

    double flm = f(lm);
    double frm = f(rm);

    double S_left = simpson_segment(fa, flm, fm, a, m);
    double S_right = simpson_segment(fm, frm, fb, m, b);

    double diff = S_left + S_right - S;

    /* 收敛判断：当误差足够小或达到最大递归深度时返回修正值 */
    if (depth <= 0 || fabs(diff) <= 15.0 * eps)
    {
        /* 使用 Richardson 外推提高精度 */
        return S_left + S_right + diff / 15.0;
    }

    /* 递归到左右子区间，误差各分配一半 */
    double left_result = adaptive_simpson_recursive(f, a, m, eps / 2.0, S_left, fa, fm, flm, depth - 1);
    double right_result = adaptive_simpson_recursive(f, m, b, eps / 2.0, S_right, fm, fb, frm, depth - 1);

    return left_result + right_result;
}

/* 主接口：自适应辛普森规则
   f: 被积函数
   a,b: 积分区间
   n: 最小分段数下限（若 n>1，将区间均分为 n 段，分别做自适应）
   eps: 绝对精度容忍度
*/
double simpsons_rule(double (*f)(double), double a, double b, int n, double eps)
{
    if (a == b)
        return 0.0;

    if (eps <= 0.0)
        eps = 1e-8; /* 防止用户传入非正值，作为保护 */

    if (n < 1)
        n = 1;

    double total = 0.0;

    /* 如果 n == 1，直接对整个区间做自适应 */
    if (n == 1)
    {
        double fa = f(a);
        double fb = f(b);
        double m = 0.5 * (a + b);
        double fm = f(m);
        double S = simpson_segment(fa, fm, fb, a, b);
        total = adaptive_simpson_recursive(f, a, b, eps, S, fa, fb, fm, MAX_RECURSION_DEPTH);
        return total;
    }

    /* 若 n > 1，将区间均分为 n 段，对每段分别应用自适应（并把 eps 平均分配到每段） */
    double h = (b - a) / (double)n;
    double eps_per_segment = eps / (double)n;

    for (int i = 0; i < n; ++i)
    {
        double x0 = a + i * h;
        double x1 = x0 + h;
        double xm = 0.5 * (x0 + x1);

        double f0 = f(x0);
        double f1 = f(x1);
        double fm = f(xm);

        double Sseg = simpson_segment(f0, fm, f1, x0, x1);

        double seg_result = adaptive_simpson_recursive(f, x0, x1, eps_per_segment, Sseg, f0, f1, fm, MAX_RECURSION_DEPTH);
        total += seg_result;
    }

    return total;
}

/* 便捷包装函数：使用默认 eps = 1e-8 */
double simpsons_rule_default(double (*f)(double), double a, double b, int n)
{
    return simpsons_rule(f, a, b, n, 1e-8);
}

/* --- 示例：f(x) = x^2 --- */
double f(double x)
{
    return x * x;
}

int main()
{
    double a, b;
    int n;
    double eps;

    printf("请输入积分的下限 a: ");
    if (scanf("%lf", &a) != 1)
        return 1;
    printf("请输入积分的上限 b: ");
    if (scanf("%lf", &b) != 1)
        return 1;
    printf("请输入最小分段数 n（整数，n<=1 表示不分段，使用整个区间）: ");
    if (scanf("%d", &n) != 1)
        return 1;
    printf("请输入精度 eps（输入 0 使用默认 1e-8）: ");
    if (scanf("%lf", &eps) != 1)
        return 1;
    if (eps == 0.0)
        eps = 1e-8;

    double result = simpsons_rule(f, a, b, n, eps);

    printf("积分结果是: %.12f\n", result);

    return 0;
}