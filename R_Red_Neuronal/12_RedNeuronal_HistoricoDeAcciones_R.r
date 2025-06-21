
# En el siguiente codigo se muestra la creacion de una red neuronal con lenguaje R
# Con perceptrones que indican las iteraciones de la red neuronal 
# Instala las siguiente librerias si no los tienes
install.packages("quantmod")
install.packages("tidyquant")
install.packages("PerformanceAnalytics")

# Carga los paquetes de las librerias
library(quantmod)
library(tidyquant)
library(PerformanceAnalytics)


# Primero se descargan datos históricos de Apple (AAPL)
getSymbols("AAPL", src = "yahoo", from = "2020-01-01", to = Sys.Date())

# Posteriormente visualizar el precio de cierre ajustado
chartSeries(AAPL, type = "line", theme = chartTheme("white"), TA = NULL)

# Procedemos a usar tidyquant con ggplot2 para organizar la grafica
library(ggplot2)
AAPL_df <- tidyquant::tq_get("AAPL", from = "2020-01-01", to = Sys.Date())
ggplot(AAPL_df, aes(x = date, y = adjusted)) +
  geom_line(color = "steelblue") +
  labs(title = "Precio Ajustado de AAPL", x = "Fecha", y = "Precio")

# Retornos diarios logarítmicos
returns <- dailyReturn(Cl(AAPL), type = "log")

# Visualizar los retornos
plot(returns, main = "Retornos diarios (log) de AAPL")

# Extraemos de la tabla los retornos mensuales
monthly_returns <- periodReturn(AAPL, period = 'monthly', type = 'log')


# Sacamos las estadísticas descriptivas
summary(returns)
sd(returns)      # Volatilidad
mean(returns)    # Rentabilidad promedio

# Drawdown
chart.Drawdown(returns)


# Continuamos con descargar varias acciones de la tabla
tickers <- c("AAPL", "MSFT", "GOOG")
data <- tq_get(tickers, from = "2020-01-01", to = Sys.Date())

# Por ultimo calculamos retornos y graficar
returns_df <- data %>%
  group_by(symbol) %>%
  tq_transmute(select = adjusted, mutate_fun = periodReturn, period = "monthly", type = "log")

# Finalizamos retornando la grafica
returns_df %>%
  ggplot(aes(x = date, y = monthly.returns, color = symbol)) +
  geom_line() +
  labs(title = "Retornos mensuales", x = "Fecha", y = "Retorno logarítmico")

