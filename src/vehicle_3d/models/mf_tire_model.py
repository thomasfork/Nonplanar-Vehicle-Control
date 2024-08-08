'''
complete Magic Formula tire model
adapted from Pacejka, Hans. Tire and vehicle dynamics. Elsevier, 2005.
'''

from dataclasses import dataclass, field

import casadi as ca

from vehicle_3d.pytypes import PythonMsg, NestedPythonMsg
from vehicle_3d.utils import ca_utils
from vehicle_3d.models.model import Model, ModelVars, ModelConfig

@dataclass
class TireDimensionConfig(ModelConfig):
    ''' configurations for tire dimensions '''
    r0: float = field(default = 0.3135)
    w: float = field(default = 0.2)
    r_rim: float = field(default = 0.1905)

@dataclass
class TireConditionConfig(ModelConfig):
    ''' configurations for tire dimensions '''
    pi0: float = field(default = 220000)

@dataclass
class TireInertiaConfig(ModelConfig):
    ''' configurations for tire condition '''
    m_tire: float = field(default = 9.3)
    Ixx_tire: float = field(default = 0.391)
    Iyy_tire: float = field(default = 0.736)
    m_belt: float = field(default = 7.247)
    Ixx_belt: float = field(default = 0.3519)
    Iyy_belt: float = field(default = 0.5698)

@dataclass
class TireVerticalConfig(ModelConfig):
    ''' configurations for tire vertical response '''
    Fz0: float = field(default = 4000)
    cz0: float = field(default = 209651)
    kz: float = field(default = 50)
    qv2: float = field(default = 0.04667)
    qFz2: float = field(default = 15.4)
    qFcx: float = field(default = 0)
    qFcy: float = field(default = 0)
    pFz1: float = field(default = 0.7098)
    Breff: float = field(default = 8.386)
    Dreff: float = field(default = 0.25826)
    Freff: float = field(default = 0.07394)
    qre0: float = field(default = 0.9974)
    qv1: float = field(default = 7.742e-4)

@dataclass
class TireStructuralConfig(ModelConfig):
    ''' configurations for tire structural response '''
    cx0: float = field(default = 358066)
    pcfx1: float = field(default = 0.17504)
    pcfx2: float = field(default = 0)
    pcfx3: float = field(default = 0)
    cy0: float = field(default = 102673)
    pcfy1: float = field(default = 0.16365)
    pcfy2: float = field(default = 0)
    pcfy3: float = field(default = 0.24993)
    cphi: float = field(default = 4795)
    pcmz1: float = field(default = 0)
    flong: float = field(default = 77.17)
    flag: float = field(default = 42.41)
    fyaw: float = field(default = 53.49)
    fwindup: float = field(default = 58.95)
    zetalong: float = field(default = 0.056)
    zetalat: float = field(default = 0.037)
    zetayaw: float = field(default= 0.007)
    zetawindup: float = field(default = 0.05)
    qbvx: float = field(default = 0.364)
    qbvtheta: float = field(default = 0.065)

@dataclass
class TireContactPatchConfig(ModelConfig):
    ''' configurations for tire contact patch '''
    qra1: float = field(default = 0.671)
    qra2: float = field(default = 0.733)
    qrb1: float = field(default = 1.059)
    qrb2: float = field(default = -1.1878)
    pls: float = field(default = 0.8335)
    pae: float = field(default = 1.471)
    pbe: float = field(default = 0.9622)
    ce: float = field(default = 1.5174)

@dataclass
class TireLongitudinalConfig(ModelConfig):
    ''' configurations for tire longitudinal behavior '''
    pCx1: float = field(default = 1.579)
    pDx1: float = field(default = 1.0422)
    pDx2: float = field(default = -0.008285)
    pDx3: float = field(default = 0)
    pEx1: float = field(default = 0.11113)
    pEx2: float = field(default = 0.3143)
    pEx3: float = field(default = 0)
    pEx4: float = field(default = 0.001719)
    pKx1: float = field(default = 21.687)
    pKx2: float = field(default = 13.728)
    pKx3: float = field(default = -0.4098)
    pHx1: float = field(default = 2.1615e-4)
    pHx2: float = field(default = 0.0011598)
    pVx1: float = field(default = 0.0)
    pVx2: float = field(default = 0.0)
    ppx1: float = field(default = -0.3485)
    ppx2: float = field(default = 0.37824)
    ppx3: float = field(default = -0.09603)
    ppx4: float = field(default = 0.06518)
    rBx1: float = field(default  = 13.046)
    rBx2: float = field(default = 9.718)
    rBx3: float = field(default = 0)
    rCx1: float = field(default = 0.9995)
    rEx1: float = field(default = -0.4404)
    rEx2: float = field(default = -0.4663)
    rHx1: float = field(default = -9.968e-5)

@dataclass
class TireOverturningConfig(ModelConfig):
    ''' configurations for tire overturning behavior '''
    qsx1: float = field(default = -0.007764)
    qsx2: float = field(default = 1.1915)
    qsx3: float = field(default = 0.013948)
    qsx4: float = field(default = 4.912)
    qsx5: float = field(default = 1.02)
    qsx6: float = field(default = 22.83)
    qsx7: float = field(default = 0.7104)
    qsx8: float = field(default = -0.023393)
    qsx9: float = field(default = 0.6581)
    qsx10: float = field(default = 0.2824)
    qsx11: float = field(default = 5.349)
    qsx12: float = field(default = 0)
    qsx13: float = field(default = 0)
    qsx14: float = field(default = 0)
    ppMx1: float = field(default = 0)

@dataclass
class TireLateralConfig(ModelConfig):
    ''' configurations for tire lateral behavior '''
    pCy1: float = field(default = 1.338)
    pDy1: float = field(default = 0.8785)
    pDy2: float = field(default = -0.06452)
    pDy3: float = field(default = 0)
    pEy1: float = field(default = -0.8057)
    pEy2: float = field(default = -0.6046)
    pEy3: float = field(default = 0.09864)
    pEy4: float = field(default= -6.697)
    pEy5: float = field(default = 0)
    pKy1: float = field(default = 15.324)
    pKy2: float = field(default = 1.715)
    pKy3: float = field(default = 0.3695)
    pKy4: float = field(default = 2.0005)
    pKy5: float = field(default = 0)
    pKy6: float = field(default = -0.8987)
    pKy7: float = field(default = -0.23303)
    pHy1: float = field(default = -0.001806)
    pHy2: float = field(default = 0.00352)
    pVy1: float = field(default = 0.0)
    pVy2: float = field(default = 0.0)
    pVy3: float = field(default = -0.162)
    pVy4: float = field(default = -0.4864)
    ppy1: float = field(default = -0.6255)
    ppy2: float = field(default = -0.06523)
    ppy3: float = field(default = -0.16666)
    ppy4: float = field(default = 0.2811)
    ppy5: float = field(default = 0)
    rBy1: float = field(default = 10.622)
    rBy2: float = field(default = 7.82)
    rBy3: float = field(default = 0.002037)
    rBy4: float = field(default = 0)
    rCy1: float = field(default = 1.0587)
    rEy1: float = field(default = 0.3148)
    rEy2: float = field(default = 0.004867)
    rHy1: float = field(default = 0.009472)
    rHy2: float = field(default = 0.009754)
    rVy1: float = field(default = 0.05187)
    rVy2: float = field(default = 4.853e-4)
    rVy3: float = field(default = 0)
    rVy4: float = field(default = 94.63)
    rVy5: float = field(default = 1.8914)
    rVy6: float = field(default = 23.8)

@dataclass
class TireRollingConfig(ModelConfig):
    ''' configurations for tire rolling behavior '''
    qsy1: float = field(default = 0.00702)
    qsy2: float = field(default = 0)
    qsy3: float = field(default = 0.001515)
    qsy4: float = field(default = 8.514e-5)
    qsy5: float = field(default = 0)
    qsy6: float = field(default = 0)
    qsy7: float = field(default = 0.9008)
    qsy8: float = field(default = -0.4089)

@dataclass
class TireAligningConfig(ModelConfig):
    ''' configurations for tire aligning behavior '''
    qBz1: float = field(default = 12.035)
    qBz2: float = field(default = -1.33)
    qBz3: float = field(default = 0)
    qBz4: float = field(default = 0.176)
    qBz5: float = field(default = -0.14853)
    qBz9: float = field(default = 34.5)
    qBz10: float = field(default = 0)
    qCz1: float = field(default = 1.2923)
    qDz1: float = field(default = 0.09068)
    qDz2: float = field(default = -0.00565)
    qDz3: float = field(default = 0.3778)
    qDz4: float = field(default = 0)
    qDz6: float = field(default = 0.0017015)
    qDz7: float = field(default = -0.002091)
    qDz8: float = field(default = -0.1428)
    qDz9: float = field(default = 0.00915)
    qDz10: float = field(default = 0)
    qDz11: float = field(default = 0)
    qEz1: float = field(default = -1.7924)
    qEz2: float = field(default = 0.8975)
    qEz3: float = field(default = 0)
    qEz4: float = field(default = 0.2895)
    qEz5: float = field(default = -0.6786)
    qHz1: float = field(default = 0.0014333)
    qHz2: float = field(default = 0.00230867)
    qHz3: float = field(default = 0.24973)
    qHz4: float = field(default = -0.21205)
    ppz1: float = field(default = -0.4408)
    ppz2: float = field(default = 0)
    ssz1: float = field(default = 0.00918)
    ssz2: float = field(default = 0.03869)
    ssz3: float = field(default = 0)
    ssz4: float = field(default = 0)

@dataclass
class TireTurnslipConfig(ModelConfig):
    ''' configurations for tire turn slip '''
    pDxphi1: float = field(default = 0.4)
    pDxphi2: float = field(default = 0)
    pDxphi3: float = field(default = 0)
    pKyphi1: float = field(default = 1)
    pDyphi1: float = field(default = 0.4)
    pDyphi2: float = field(default = 0)
    pDyphi3: float = field(default = 0)
    pDyphi4: float = field(default = 0)
    pHyphi1: float = field(default = 1)
    pHyphi2: float = field(default = 0.15)
    pHyphi3: float = field(default = 0)
    pHyphi4: float = field(default = -4)
    pEyphi1: float = field(default = 0.5)
    pEyphi2: float = field(default = 0)
    qDtphi1: float = field(default = 10)
    qCrphi1: float = field(default = 0.2)
    qCrphi2: float = field(default = 0.1)
    qBrphi1: float = field(default = 0.1)
    qDrphi1: float = field(default = 1)
    qDrphi2: float = field(default = 0)

@dataclass
class TireScalingConfig(ModelConfig):
    ''' configurations for tire scaling '''
    lambda_Fz0: float = field(default = 1)
    lambda_mux: float = field(default = 1)
    lambda_muy: float = field(default = 1)
    lambda_muV: float = field(default = 0)
    lambda_Kxk: float = field(default = 1)
    lambda_Kya: float = field(default = 1)
    lambda_Cx: float = field(default  = 1)
    lambda_Cy: float = field(default  = 1)
    lambda_Ex: float = field(default  = 1)
    lambda_Ey: float = field(default  = 1)
    lambda_Hx: float = field(default  = 1)
    lambda_Hy: float = field(default  = 1)
    lambda_Vx: float = field(default  = 1)
    lambda_Vy: float = field(default  = 1)
    lambda_Kygamma: float = field(default  = 1)
    lambda_Kzgamma: float = field(default  = 1)
    lambda_t: float = field(default  = 1)
    lambda_Mr: float = field(default  = 1)
    lambda_xa: float = field(default  = 1)
    lambda_yk: float = field(default  = 1)
    lambda_Vyk: float = field(default  = 1)
    lambda_s: float = field(default  = 1)
    lambda_Cz: float = field(default  = 1)
    lambda_Mx: float = field(default  = 1)
    lambda_VMx: float = field(default  = 1)
    lambda_My: float = field(default  = 1)
    lambda_Mphi: float = field(default  = 1)

@dataclass
class TireConfig(ModelConfig, NestedPythonMsg):
    ''' configuration parameters of a tire '''
    V0: float = field(default = 16.7)
    V_low: float = field(default = 1.0)
    dimension: TireDimensionConfig = field(default=None)
    condition: TireConditionConfig = field(default=None)
    inertia: TireInertiaConfig = field(default=None)
    vertical: TireVerticalConfig = field(default=None)
    structural: TireStructuralConfig = field(default=None)
    contact: TireContactPatchConfig = field(default=None)
    longitudinal: TireLongitudinalConfig = field(default=None)
    overturning: TireOverturningConfig = field(default=None)
    lateral: TireLateralConfig = field(default=None)
    rolling: TireRollingConfig = field(default=None)
    alinging: TireAligningConfig = field(default=None)
    turnslip: TireTurnslipConfig = field(default=None)
    scaling: TireScalingConfig = field(default=None)

@dataclass
class TireState(PythonMsg):
    ''' state of a tire '''
    s: float = field(default = 0.) # slip ratio
    a: float = field(default = 0.) # slip angle
    y: float = field(default = 0.) # camber angle
    p: float = field(default = 0.) # turn slip
    Fx: float = field(default = 0.) # longitudinal force
    Fy: float = field(default = 0.) # lateral force
    Fz: float = field(default = 0.) # vertical force
    Mx: float = field(default = 0.) # longitudinal moment
    My: float = field(default = 0.) # lateral moment
    Mz: float = field(default = 0.) # vertical moment

@dataclass
class TireModelVars(ModelVars):
    ''' tire model variables '''


class TireModel(Model):
    ''' magic formula tire model '''
    config: TireConfig
    model_vars: TireModelVars

    f_Ft: ca.Function
    f_Kt: ca.Function

    def __init__(self, config: TireConfig):
        self.config = config
        self.model_vars = TireModelVars()
        self.setup()

    def setup(self):
        ''' set up the model to match current config '''
        alpha = ca.SX.sym('a') # slip angle
        kappa = ca.SX.sym('k') # slip ratio
        gamma = ca.SX.sym('y') # camber angle
        phi   = ca.SX.sym('phi') # turn slip
        Fz    = ca.SX.sym('Fz') # normal force on the tire
        pi    = ca.SX.sym('pi') # tire inflation pressure
        Vcx   = ca.SX.sym('Vcx') # wheel center velocity, only need its sign
        tire_inputs = [kappa, alpha, gamma, phi, Fz, pi, Vcx]

        # modified variables
        Fz0_prime = self.config.scaling.lambda_Fz0 * self.config.vertical.Fz0
        dfz = (Fz - Fz0_prime)/Fz0_prime
        dpi = (pi - self.config.condition.pi0) / self.config.condition.pi0
        gamma_star = ca.sin(gamma)
        alpha_star = ca.tan(alpha) * ca.sign(Vcx)

        #small nonzero numbers for pacejka model
        eps_x = 1e-9
        eps_K = 1e-9
        eps_y = 1e-9
        eps_r = 1e-9

        # longitudinal and lateral turnslip coefficients
        zeta_0 = 0
        Bxphi = self.config.turnslip.pDxphi1 * (1 + self.config.turnslip.pDxphi2*dfz) \
            * ca.cos(ca.arctan(self.config.turnslip.pDxphi3 * kappa))
        zeta_1 = ca.cos(ca.arctan(Bxphi * self.config.dimension.r0 * phi))
        Byphi = self.config.turnslip.pDyphi1 * (1 + self.config.turnslip.pDyphi2 * dfz) \
            * ca.cos(ca.arctan(self.config.turnslip.pDyphi3 * ca.tan(alpha)))
        zeta_2 = ca.cos(ca.arctan(Byphi*self.config.dimension.r0 * ca_utils.ca_abs(phi)\
            + self.config.turnslip.pDyphi4 * ca.sqrt(self.config.dimension.r0 \
            * ca_utils.ca_abs(phi))))
        zeta_3 = ca.cos(ca.arctan(self.config.turnslip.pKyphi1*self.config.dimension.r0**2 \
            * phi**2))
        eps_gamma = self.config.turnslip.pEyphi1*(1 + self.config.turnslip.pEyphi2*dfz)
        zeta_5 = ca.cos(ca.arctan(self.config.turnslip.qDtphi1\
            * self.config.dimension.r0 * phi))
        zeta_6 = ca.cos(ca.arctan(self.config.turnslip.qBrphi1\
            * self.config.dimension.r0 * phi))

        # pure longitudinal force
        SVx = Fz * (self.config.longitudinal.pVx1 + self.config.longitudinal.pVx2*dfz)\
            *self.config.scaling.lambda_Vx * self.config.scaling.lambda_mux * zeta_1
        SHx =      (self.config.longitudinal.pHx1 + self.config.longitudinal.pHx2*dfz)\
            *self.config.scaling.lambda_Hx
        kappa_x = kappa + SHx
        mu_x = (self.config.longitudinal.pDx1 + self.config.longitudinal.pDx2*dfz)\
            *(1+self.config.longitudinal.ppx3*dpi * self.config.longitudinal.ppx4*dpi*dpi)\
            *(1-self.config.longitudinal.pDx3*gamma**2)*self.config.scaling.lambda_mux
        Cx = self.config.longitudinal.pCx1 * self.config.scaling.lambda_Cx
        Cx = ca_utils.ca_pos(Cx)
        Dx = mu_x * Fz * zeta_1
        Dx = ca_utils.ca_pos(Dx)
        Kxk = Fz*(self.config.longitudinal.pKx1 + self.config.longitudinal.pKx2*dfz) \
            * ca.exp(self.config.longitudinal.pKx3*dfz)\
            *(1 + self.config.longitudinal.ppx1*dpi + self.config.longitudinal.ppx2 * dpi**2)
        Ex = (self.config.longitudinal.pEx1 + self.config.longitudinal.pEx2*dfz \
              + self.config.longitudinal.pEx3*dfz**2)\
            *(1-self.config.longitudinal.pEx4*ca.sign(kappa_x)) * self.config.scaling.lambda_Ex
        Ex = ca_utils.ca_leq(Ex, 1)
        Bx = Kxk / (Cx*Dx + eps_x)
        Fx0 = Dx*ca.sin(Cx*ca.arctan(Bx*kappa_x - Ex*(Bx*kappa_x - ca.arctan(Bx*kappa_x)))) + SVx

        # pure lateral force
        Kygamma0 = Fz*(self.config.lateral.pKy6 + self.config.lateral.pKy7*dfz) \
            * (1+self.config.lateral.ppy5*dpi)*self.config.scaling.lambda_Kygamma
        SVygamma = Fz*(self.config.lateral.pVy3 + self.config.lateral.pVy4*dfz) \
            * gamma_star \
            * self.config.scaling.lambda_Kygamma * self.config.scaling.lambda_muy * zeta_2
        SVy = Fz*(self.config.lateral.pVy1 + self.config.lateral.pVy2*dfz) \
            * self.config.scaling.lambda_Vy * self.config.scaling.lambda_muy \
            * zeta_2 + SVygamma
        Kya = self.config.lateral.pKy1 * Fz0_prime * (1 + self.config.lateral.ppy1*dpi) \
            * (1 - self.config.lateral.pKy1 * ca_utils.ca_abs(gamma_star)) \
            * ca.sin(self.config.lateral.pKy4 \
                * ca.arctan((Fz / Fz0_prime) / \
                ((self.config.lateral.pKy2 + self.config.lateral.pKy5*gamma_star **2) \
                 * (1 + self.config.lateral.ppy2 * dpi)))) \
            * zeta_3 * self.config.scaling.lambda_Kya
        SHy = (self.config.lateral.pHy1 + self.config.lateral.pHy2 * dfz)\
            * self.config.scaling.lambda_Hy + (Kygamma0 * gamma_star - SVygamma) / (Kya + eps_K)
        alpha_y = alpha_star + SHy
        mu_y = (self.config.lateral.pDy1 + self.config.lateral.pDy2*dfz) \
            * (1 + self.config.lateral.ppy3*dpi + self.config.lateral.ppy4*dpi**2) \
            * (1 - self.config.lateral.pDy3 * gamma_star ** 2) * self.config.scaling.lambda_muy
        Ey = (self.config.lateral.pEy1 + self.config.lateral.pEy2*dfz) \
            * (1 + self.config.lateral.pEy5*gamma_star**2 - \
                (self.config.lateral.pEy3 + self.config.lateral.pEy4*gamma_star)*ca.sign(alpha_y)) \
            * self.config.scaling.lambda_Ey
        Ey = ca_utils.ca_leq(Ey, 1)
        Dy = mu_y * Fz * zeta_2
        Cy = self.config.lateral.pCy1 * self.config.scaling.lambda_Cy
        Cy = ca_utils.ca_pos(Cy)
        By = Kya / (Cy*Dy + eps_y)
        Fy0 = Dy*ca.sin(Cy*ca.arctan(By*alpha_y - Ey*(By*alpha_y - ca.arctan(By*alpha_y)))) + SVy

        # combined longitudinal force
        SHxa = self.config.longitudinal.rHx1
        Exa = self.config.longitudinal.rEx1 + self.config.longitudinal.rEx2*dfz
        Cxa = self.config.longitudinal.rCx1
        Bxa = (self.config.longitudinal.rBx1 + self.config.longitudinal.rBx3*gamma_star**2) \
            * ca.cos(ca.arctan(self.config.longitudinal.rBx2*kappa)) \
            * self.config.scaling.lambda_xa
        alpha_s = alpha_star + SHxa
        Gxa0 = ca.cos(Cxa*ca.arctan(Bxa*SHxa - Exa*(Bxa*SHxa - ca.arctan(Bxa*SHxa))))
        Gxa  = ca.cos(Cxa*ca.arctan(Bxa*alpha_s - Exa*(Bxa*alpha_s - ca.arctan(Bxa*alpha_s)))) \
            / Gxa0
        Fx = Gxa * Fx0

        # combined lateral force
        Byk = (self.config.lateral.rBy1 + self.config.lateral.rBy4 * gamma_star **2) \
            * ca.cos(ca.arctan(self.config.lateral.rBy2*(alpha_star - self.config.lateral.rBy3))) \
            * self.config.scaling.lambda_yk
        Cyk = self.config.lateral.rCy1
        Eyk = self.config.lateral.rEy1 + self.config.lateral.rEy2 * dfz
        SHyk = self.config.lateral.rHy1 + self.config.lateral.rHy2 * dfz
        DVyk = mu_y * Fz * (self.config.lateral.rVy1 + self.config.lateral.rVy2*dfz \
            + self.config.lateral.rVy3*gamma_star) \
            * ca.cos(ca.arctan(self.config.lateral.rVy4 * alpha_star)) * zeta_2
        SVyk = DVyk * ca.sin(self.config.lateral.rVy5 \
                             * ca.arctan(self.config.lateral.rVy6 * kappa)) \
            * self.config.scaling.lambda_Vyk
        kappa_s = kappa + SHyk
        Gyk0 = ca.cos(Cyk*ca.arctan(Byk*SHyk - Eyk*(Byk*SHyk - ca.arctan(Byk*SHyk))))
        Gyk = ca.cos(Cyk*ca.arctan(Byk*kappa_s - Eyk*(Byk*kappa_s - ca.arctan(Byk*kappa_s)))) \
            / Gyk0
        Fy = Gyk * Fy0 + SVyk

        # aligning torque turnslip coefficients
        Mzphiinf = self.config.turnslip.qCrphi1*mu_y*self.config.dimension.r0 \
            * Fz * ca.sqrt(Fz / Fz0_prime) * self.config.scaling.lambda_Mphi
        CDrphi = self.config.turnslip.qDrphi1
        EDrphi = self.config.turnslip.qDrphi2
        DDrphi = Mzphiinf / ca.sin(0.5*ca.pi * CDrphi)
        Kzyr0 = Fz * self.config.dimension.r0 \
            * (self.config.alinging.qDz8 + self.config.alinging.qDz9*dfz \
            + (self.config.alinging.qDz10 + self.config.alinging.qDz11*dfz) * ca_utils.ca_abs(gamma)
            ) * self.config.scaling.lambda_Kzgamma
        BDrphi = Kzyr0 / (CDrphi * DDrphi * (1 - eps_gamma) + eps_r)
        Drphi = DDrphi * ca.sin(CDrphi * ca.arctan(BDrphi * self.config.dimension.r0*phi \
            - EDrphi*(BDrphi*self.config.dimension.r0*phi \
            - ca.arctan(BDrphi * self.config.dimension.r0 * phi)) ))
        Mzphi90 = Mzphiinf * 2 / ca.pi * ca.arctan(self.config.turnslip.qCrphi2 \
            * self.config.dimension.r0 * ca_utils.ca_abs(phi)) * Gyk
        zeta_7 = 2 / ca.pi * ca.arccos(Mzphi90 / (
            ca_utils.ca_abs(Drphi) + eps_r
        ))

        zeta_8 = 1 + Drphi

        # pure aligning torque
        SHt = self.config.alinging.qHz1 + self.config.alinging.qHz2*dfz \
            + (self.config.alinging.qHz3 + self.config.alinging.qHz4*dfz) * gamma_star
        alpha_t = alpha_star + SHt
        Dt0 = Fz * (self.config.dimension.r0/Fz0_prime) \
            * (self.config.alinging.qDz1 + self.config.alinging.qDz2*dfz)\
            * (1-self.config.alinging.ppz1 * dpi) \
            * self.config.scaling.lambda_t * ca.sign(Vcx)
        Br = (self.config.alinging.qBz9*self.config.scaling.lambda_Kya\
            /self.config.scaling.lambda_muy + self.config.alinging.qBz10*By*Cy)*zeta_6
        Cr = zeta_7
        Dr = Fz*self.config.dimension.r0 \
            * ((self.config.alinging.qDz6 + self.config.alinging.qDz9*dfz)\
            * self.config.scaling.lambda_Mr*zeta_2 \
            + ((self.config.alinging.qDz8 + self.config.alinging.qDz9*dfz)\
            * (1+self.config.alinging.ppz2*dpi) \
            + (self.config.alinging.qDz10+self.config.alinging.qDz11*dfz)\
            * ca_utils.ca_abs(gamma_star)) * \
        gamma_star * self.config.scaling.lambda_Kzgamma * zeta_0 ) \
            * self.config.scaling.lambda_muy * ca.sign(Vcx) * ca.cos(alpha) + zeta_8 - 1
        Bt = (self.config.alinging.qBz1 + self.config.alinging.qBz2*dfz \
            + self.config.alinging.qBz3*dfz**2)\
            * (1+self.config.alinging.qBz5 * ca_utils.ca_abs(gamma_star) \
            + self.config.alinging.qBz9 * gamma_star**2) \
            * self.config.scaling.lambda_Kya / self.config.scaling.lambda_muy  #book put qbz6
        Ct = self.config.alinging.qCz1
        Dt = Dt0 * (1+self.config.alinging.qDz3 * ca_utils.ca_abs(gamma_star)\
            + self.config.alinging.qDz4 * gamma_star **2) * zeta_5
        Et = (self.config.alinging.qEz1 + self.config.alinging.qEz2*dfz\
            + self.config.alinging.qEz3*dfz**2) \
            * (1 + (self.config.alinging.qEz4 + self.config.alinging.qEz5*gamma_star)\
            * 2/ca.pi*ca.arctan(Bt*Ct*alpha_t))
        Kya_prime = Kya + eps_K
        SHf = SHy + SVy / Kya_prime
        alpha_r = alpha_star + SHf
        # Mzr0 = Dr * ca.cos(Cr * ca.arctan(Br * alpha_r)) * ca.cos(alpha)
        # t0 = Dt * ca.cos(Ct * ca.arctan(Bt*alpha_t - Et*(Bt*alpha_t - ca.arctan(Bt*alpha_t))))
        # Mz0_prime = -t0 *ca.substitute(Fy0,gamma,0)
        # Mz0 = Mz0_prime + Mzr0

        #aligning torque (combined slip)
        alpha_teq = ca.sqrt(alpha_t**2 + (Kxk/Kya_prime)**2 *kappa**2) * ca.sign(alpha_t)
        alpha_req = ca.sqrt(alpha_r**2 + (Kxk/Kya_prime)**2 *kappa**2) * ca.sign(alpha_r)
        s = self.config.dimension.r0 \
            * (self.config.alinging.ssz1 + self.config.alinging.ssz2*(Fy/Fz0_prime) \
               + (self.config.alinging.ssz3 + self.config.alinging.ssz4*dfz)*gamma_star) \
            * self.config.scaling.lambda_s
        Mzr = Dr*ca.cos(Cr * ca.arctan(Br * alpha_req))
        Fy_prime = ca.substitute(Gyk, gamma, 0) * ca.substitute(Fy0, gamma, 0)
        t = Dt * ca.cos(Ct*ca.arctan(Bt*alpha_teq - Et*(Bt*alpha_teq - ca.arctan(Bt*alpha_teq)))) \
            * ca.cos(alpha)
        Mz_prime = -t*Fy_prime
        Mz = Mz_prime + Mzr + s * Fx

        # Overturning couple
        Mx = self.config.dimension.r0 * Fz \
            * (self.config.overturning.qsx1*self.config.scaling.lambda_VMx \
            - self.config.overturning.qsx2*gamma*(1+self.config.overturning.ppMx1*dpi) \
            + self.config.overturning.qsx3 * Fy/self.config.vertical.Fz0 +
            self.config.overturning.qsx4*ca.cos(self.config.overturning.qsx5\
            * ca.arctan(self.config.overturning.qsx6*Fz/self.config.vertical.Fz0)**2) \
            * ca.sin(self.config.overturning.qsx7*gamma + self.config.overturning.qsx8\
            * ca.arctan(self.config.overturning.qsx9*Fy/self.config.vertical.Fz0)) +
            self.config.overturning.qsx10 * ca.arctan(self.config.overturning.qsx11\
            * Fz/self.config.vertical.Fz0)*gamma)*self.config.scaling.lambda_Mx

        # Rolling resistance moment
        My = Fz*self.config.dimension.r0 \
            * (self.config.rolling.qsy1 + self.config.rolling.qsy2*Fx/self.config.vertical.Fz0 \
            + self.config.rolling.qsy3 * ca_utils.ca_abs(Vcx/self.config.V0) \
            + self.config.rolling.qsy4*(Vcx/self.config.V0)**4 +
            (self.config.rolling.qsy5 + self.config.rolling.qsy6*Fz/self.config.vertical.Fz0)\
            * gamma**2) *(ca.power(Fz/self.config.vertical.Fz0, self.config.rolling.qsy7) \
            * ca.power(pi/self.config.condition.pi0, self.config.rolling.qsy7)) \
            * self.config.scaling.lambda_My

        outputs = ca.vertcat(Fx, Fy, Fz, Mx, My, Mz)

        self.model_vars.u = tire_inputs
        self.model_vars.g = outputs
        self.f_Ft = ca.Function('Ft',tire_inputs,[ca.vertcat(Fx, Fy, Fz)])
        self.f_Kt = ca.Function('Kt',tire_inputs,[ca.vertcat(Mx, My, Mz)])

    def get_empty_state(self) -> TireState:
        return TireState()
