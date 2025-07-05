import React from 'react';
import { cn } from '../../lib/utils';

export const buttonVariants = {
  base: 'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none',
  primary: 'bg-blue-600 text-white hover:bg-blue-500',
};

export const Button = React.forwardRef(function Button({ className = '', variant = 'primary', ...props }, ref) {
  const classes = cn(buttonVariants.base, buttonVariants[variant], className);
  return <button ref={ref} className={classes} {...props} />;
});
